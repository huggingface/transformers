"""Implementation of Conditional ViTPlus detection model.

The implementation allows for: 1) using label-embeddings to use as fixed class
projection, 2) (optionally) conditioning the decoder on a set of given labels.
"""
from absl import logging
import functools
from typing import Sequence, Any, Dict, List, Mapping, Optional, Callable, Tuple

import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import utils
from clip_files import model as clip_model
from clip_files import tokenizer as clip_tokenizer



# Match PyTorch default LayerNorm epsilon of 1e-5 (FLAX defaults to 1e-6).
LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-5)


def quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
  return x * jax.nn.sigmoid(1.702 * x)


class Shortcut(nn.Module):
  """Shortcut in ResNet.

  Attributes:
    features: Number of features.
    stride: Stride of the down-sampled output.
  """
  features: int
  stride: int

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = nn.avg_pool(x, (self.stride, self.stride), (self.stride, self.stride))
    x = nn.Conv(
        self.features, (1, 1), strides=(1, 1), use_bias=False, name='0')(x)
    x = nn.BatchNorm(use_running_average=True, name='1')(x)
    return x


class Bottleneck(nn.Module):
  """Bottleneck layer of ResNet.

  Attributes:
    features: Number of features.
    stride: Stride of the down-sampled output.
    expansion: Expansion of feature dimension.
  """
  features: int
  stride: int = 1
  expansion: int = 4

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    conv1 = nn.Conv(self.features, (1, 1), use_bias=False, name='conv1')
    bn1 = nn.BatchNorm(use_running_average=True, name='bn1')

    conv2 = nn.Conv(self.features, (3, 3), padding=[(1, 1), (1, 1)],
                    use_bias=False, name='conv2')
    bn2 = nn.BatchNorm(use_running_average=True, name='bn2')

    conv3 = nn.Conv(
        self.features * self.expansion, (1, 1), use_bias=False, name='conv3')
    bn3 = nn.BatchNorm(use_running_average=True, name='bn3')

    out = nn.relu(bn1(conv1(x)))
    out = nn.relu(bn2(conv2(out)))
    out = nn.avg_pool(out, (self.stride, self.stride),
                      (self.stride, self.stride))
    out = bn3(conv3(out))

    downsample = self.stride > 1 or x.shape[-1] != self.features * self.expansion
    if downsample:
      x = Shortcut(features=self.features * self.expansion,
                   stride=self.stride, name='downsample')(x)

    out += x
    out = nn.relu(out)
    return out


class AttentionPool(nn.Module):
  """Attention pooling layer.

  Attributes:
    num_heads: Number of heads.
    features: Number of features.
  """
  num_heads: int
  features: Optional[int] = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape(x.shape[0], -1, x.shape[3])

    x = jnp.concatenate([x.mean(axis=1, keepdims=True), x], axis=1)

    positional_embedding = self.param(
        'positional_embedding',
        jax.nn.initializers.normal(1. / x.shape[-1]**0.5),
        (x.shape[1], x.shape[2]))
    attn = nn.MultiHeadDotProductAttention(
        self.num_heads,
        qkv_features=x.shape[-1],
        use_bias=True,
        out_features=self.features,
        name='attn')

    x = x + positional_embedding[jnp.newaxis].astype(x.dtype)
    x = attn(x[:, :1], x)
    return x[:, 0]


class ResNetStage(nn.Module):
  """Attention pooling layer.

  Attributes:
    features: Number of features.
    num_layers: Number of bottleneck blocks.
    stride: Stride in the Bottleneck module.
  """
  features: int
  num_layers: int
  stride: int = 1

  @nn.compact
  def __call__(self, x: jnp.array) -> jnp.ndarray:
    x = Bottleneck(self.features, self.stride, name='0')(x)
    for i in range(1, self.num_layers):
      x = Bottleneck(self.features, name=str(i))(x)
    return x


class ModifiedResNet(nn.Module):
  """A ResNet class that is similar to torchvision's with changes.

  - There are now 3 "stem" convolutions as opposed to 1, with an average pool
  instead of a max pool.
  - Performs anti-aliasing strided convolutions, where an avgpool is
  prepended to convolutions with stride > 1 - The final pooling layer is a
  QKV attention instead of an average pool.

  Attributes:
    features: Number of features.
    out_features: Number of output features. If None, return resnet feature-map.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
  """
  features: int
  out_features: Optional[int]
  num_layers: Sequence[int]
  num_heads: Optional[int]

  def setup(self):
    # The 3-layer stem.
    self.conv1 = nn.Conv(
        self.features // 2,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        name='conv1')
    self.bn1 = nn.BatchNorm(use_running_average=True, name='bn1')
    self.conv2 = nn.Conv(
        self.features // 2,
        kernel_size=(3, 3),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        name='conv2')
    self.bn2 = nn.BatchNorm(use_running_average=True, name='bn2')
    self.conv3 = nn.Conv(
        self.features,
        kernel_size=(3, 3),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        name='conv3')
    self.bn3 = nn.BatchNorm(use_running_average=True, name='bn3')

    # Residual layers.
    self.layer1 = ResNetStage(self.features, self.num_layers[0], name='layer1')
    self.layer2 = ResNetStage(
        self.features * 2, self.num_layers[1], stride=2, name='layer2')
    self.layer3 = ResNetStage(
        self.features * 4, self.num_layers[2], stride=2, name='layer3')
    self.layer4 = ResNetStage(
        self.features * 8, self.num_layers[3], stride=2, name='layer4')
    if self.out_features is not None:
      self.attnpool = AttentionPool(
          self.num_heads, self.out_features, name='attnpool')

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

    def stem(x):
      for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                       (self.conv3, self.bn3)]:
        x = nn.relu(bn(conv(x)))
      x = nn.avg_pool(x, (2, 2), (2, 2))
      return x

    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = feature_map = self.layer4(x)

    if self.out_features is not None:
      x = self.attnpool(x)

    return x, feature_map


class MLP(nn.Module):
  """Simple MLP for Transformer."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    ch = x.shape[-1]
    x = nn.Dense(4 * ch, name='c_fc')(x)
    x = quick_gelu(x)
    x = nn.Dense(ch, name='c_proj')(x)
    return x


class ResidualAttentionBlock(nn.Module):
  """Self-attention block of Transformer.

  Attributes:
    num_heads: Number of heads.
    droplayer_p: Layer drop probability.
  """
  num_heads: int
  droplayer_p: float = 0.0

  def get_drop_pattern(self, x, deterministic):
    """Get drop pattern for drop layer."""
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      attn_mask: Optional[jnp.ndarray] = None,
      *,
      deterministic: bool = True) -> jnp.ndarray:
    xn = LayerNorm(name='ln_1')(x)
    y = nn.SelfAttention(
        self.num_heads, name='attn', deterministic=deterministic)(xn, attn_mask)

    # Droplayer.
    drop_pattern = self.get_drop_pattern(y, deterministic)
    x = y * (1.0 - drop_pattern) + x

    xn = LayerNorm(name='ln_2')(x)
    y = MLP(name='mlp')(xn)

    # Droplayer.
    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = y * (1.0 - drop_pattern) + x
    return x


class Transformer(nn.Module):
  """Transformer module.

  Attributes:
    features: Number of features.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
    stochastic_droplayer_rate: Stochastic depth droplayer rate.
  """
  features: int
  num_layers: int
  num_heads: int
  stochastic_droplayer_rate: float = 0.0

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               attn_mask: Optional[jnp.ndarray] = None,
               *,
               deterministic: bool = True) -> jnp.ndarray:
    for i in range(self.num_layers):
      droplayer_p = (
          i / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
      x = ResidualAttentionBlock(
          num_heads=self.num_heads,
          droplayer_p=droplayer_p,
          name=f'resblocks.{i}')(x, attn_mask, deterministic=deterministic)
    return x


class VisionTransformer(nn.Module):
  """Vision Transformer.

  Attributes:
    patch_size: The size of the patches to embed.
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    out_features: Number of output features. If None, return transformer output.
    stochastic_droplayer_rate: Stochastic depth rate.
  """
  patch_size: int
  features: int
  num_layers: int
  num_heads: int
  out_features: Optional[int]
  stochastic_droplayer_rate: float = 0.0

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               attn_mask: Optional[jnp.ndarray] = None,
               *,
               deterministic: bool = True) -> jnp.ndarray:
    x = nn.Conv(self.features,
                kernel_size=(self.patch_size, self.patch_size),
                strides=(self.patch_size, self.patch_size),
                use_bias=False, name='conv1')(x)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    scale = 1.0 / jnp.sqrt(self.features)
    class_embedding = self.param('class_embedding',
                                 jax.nn.initializers.normal(stddev=scale),
                                 (self.features,))
    x = jnp.concatenate((jnp.tile(class_embedding[None, None, :],
                                  (x.shape[0], 1, 1)), x),
                        axis=1)
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.normal(stddev=scale),
                                      (x.shape[1], self.features))
    x = x + positional_embedding[None]

    x = LayerNorm(name='ln_pre')(x)
    x = feature_map = Transformer(
        features=self.features,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        name='transformer')(
            x,
            deterministic=deterministic)

    if self.out_features is not None:
      x = LayerNorm(name='ln_post')(x[:, 0])
      x = nn.Dense(self.out_features, use_bias=False, name='proj')(x)
    else:
      x = LayerNorm(name='ln_post')(x)

    return x, feature_map


class TextEncoder(nn.Module):
  """Text Transformer.

  Attributes:
    vocab_size: Size of the vocabulary.
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    out_features: Size of the final text embedding.
  """
  vocab_size: int
  features: int
  num_layers: int
  num_heads: int
  out_features: int
  stochastic_droplayer_rate: float = 0.0

  @nn.compact
  def __call__(
      self, text: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.zeros,
                                      (text.shape[1], self.features))
    mask = nn.combine_masks(
        nn.make_attention_mask(text > 0, text > 0), nn.make_causal_mask(text))
    x = nn.Embed(self.vocab_size, self.features, name='token_embedding')(text)
    x = x + positional_embedding[None]
    x = Transformer(
        self.features,
        self.num_layers,
        self.num_heads,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        name='transformer')(
            x,
            attn_mask=mask,
            deterministic=deterministic)
    x = LayerNorm(name='ln_final')(x)
    x = x[jnp.arange(x.shape[0]), text.argmax(-1)]
    x = nn.Dense(self.out_features, use_bias=False, name='text_projection')(x)
    return x


class CLIP(nn.Module):
  """Clip model consisting of a vision and text transformer.

  Attributes:
    vocab_size: Size of the vocabulary.
    embed_dim: Size of the text and vision embeddings.
    text_features: Number of features in text transformer.
    text_num_layers: Number of text transformer blocks (self-attn + MLP).
    text_num_heads: Number of heads in text transformer.
    vision_features: Number of features in vision transformer.
    vision_num_layers: Number of vision transformer blocks (self-attn + MLP).
    vision_patch_size: Size of patches to embed in vision transformer.
  """
  vocab_size: int
  embed_dim: int
  # Text.
  text_features: int
  text_num_layers: int
  text_num_heads: int
  # Vision.
  vision_features: int
  vision_num_layers: Union[int, Sequence[int]]
  vision_patch_size: Optional[int] = None
  vision_return_map: bool = False
  # Stochastic depth.
  text_stochastic_droplayer_rate: float = 0.0
  vision_stochastic_droplayer_rate: float = 0.0

  def setup(self):
    if isinstance(self.vision_num_layers, (tuple, list)):
      self.vision_num_heads = self.vision_features * 32 // 64
      if self.vision_stochastic_droplayer_rate > 0.0:
        raise ValueError('ResNet backbone does not support stochastic depth.')
      self.visual = ModifiedResNet(
          num_layers=self.vision_num_layers,
          features=self.vision_features,
          num_heads=self.vision_num_heads,
          out_features=None if self.vision_return_map else self.embed_dim)
    else:
      self.vision_num_heads = self.vision_features // 64
      self.visual = VisionTransformer(
          patch_size=self.vision_patch_size,
          features=self.vision_features,
          num_layers=self.vision_num_layers,
          num_heads=self.vision_num_heads,
          out_features=None if self.vision_return_map else self.embed_dim,
          stochastic_droplayer_rate=self.vision_stochastic_droplayer_rate)
    self.text = TextEncoder(
        out_features=self.embed_dim,
        vocab_size=self.vocab_size,
        features=self.text_features,
        num_layers=self.text_num_layers,
        num_heads=self.text_num_heads,
        stochastic_droplayer_rate=self.text_stochastic_droplayer_rate)
    self.logit_scale = self.param('logit_scale', jax.nn.initializers.zeros, ())

  def encode_image(self,
                   image: jnp.ndarray,
                   normalize: bool = True,
                   *,
                   deterministic: bool = True) -> jnp.ndarray:
    x = self.visual(image, deterministic=deterministic)[0]
    if normalize:
      x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x

  def encode_text(self,
                  text: jnp.ndarray,
                  normalize: bool = True,
                  *,
                  deterministic: bool = True) -> jnp.ndarray:
    x = self.text(text, deterministic=deterministic)
    if normalize:
      x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x

  def __call__(self,
               image: jnp.ndarray,
               text: jnp.ndarray,
               normalize: bool = True,
               *,
               deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = y = None
    if image is not None:
      x = self.encode_image(image, normalize, deterministic=deterministic)
    if text is not None:
      y = self.encode_text(text, normalize, deterministic=deterministic)
    return x, y


class PredictorMLP(nn.Module):
  """FFN block for predicting bounding box coordinates.

  Attributes:
    out_dim: Size of output of this mlp.
    num_layers: Number of layers.
    mlp_dim: Size of hidden dimension of dense layers.
    hidden_activation: Activation function of hidden layers.
    out_activation: Activation of the output.
    dtype: Data type, e.g. jnp.float32.
  """
  out_dim: int
  num_layers: int = 1
  mlp_dim: Optional[int] = None
  hidden_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = nn.gelu
  out_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies FFN MLP block to inputs for prediction."""
    x = inputs
    mlp_dim = self.mlp_dim or x.shape[-1]
    for _ in range(self.num_layers-1):
      x = nn.Dense(mlp_dim, dtype=self.dtype)(x)
      if self.hidden_activation is not None:
        x = self.hidden_activation(x)

    x = nn.Dense(self.out_dim, kernel_init=nn.zeros)(x)
    if self.out_activation is not None:
      x = self.out_activation(x)  # pylint: disable=not-callable
    return x


class ClassPredictor(nn.Module):
  """Zero-shot instance class predictor."""
  normalize: bool = False
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      query_embeddings: Optional[jnp.ndarray] = None,
      query_mask: Optional[jnp.ndarray] = None,
  ) -> Dict[str, jnp.ndarray]:
    """Computes class prediction logits.

    Args:
      x: Image features [batch_size, num_patches, emb_dim].
      query_embeddings: The embeddings to classify against of shape [batch_size,
        num_queries, out_dim]. If not specified, only the image class embeddings
        will be returned.
      query_mask: Mask indicating whether query is real (1) or padding (0), of
        shape [batch_size, num_queries].
    Returns:
      Dict with keys 'class_embeddings' and, if query embeddings were provided,
      'pred_logits'.
    """
    if self.out_dim is not None:
      out_dim = self.out_dim
    elif query_embeddings is not None:
      out_dim = query_embeddings.shape[-1]
    else:
      raise ValueError('Unable to infer class head shape. Please pass out_dim.')

    image_class_emb = nn.Dense(
        out_dim, kernel_init=nn.initializers.normal(1e-6))(x)
    if query_embeddings is None:
      return {'class_embeddings': image_class_emb}
    assert out_dim == query_embeddings.shape[-1]

    if self.normalize:
      image_class_emb /= jnp.linalg.norm(
          image_class_emb, axis=-1, keepdims=True) + 1e-6
      query_embeddings /= jnp.linalg.norm(
          query_embeddings, axis=-1, keepdims=True) + 1e-6

    assert query_embeddings.ndim > 2, ('Expects shape (batch, query, out_dim). '
                                       f'Got {query_embeddings.shape}')
    pred_logits = jnp.einsum(
        '...pd,...qd->...pq', image_class_emb, query_embeddings)

    # Apply a learnable shift and scale to logits:
    logit_shift = nn.Dense(1, name='logit_shift')(x)
    logit_scale = nn.Dense(1, use_bias=True, name='logit_scale')(x)
    logit_scale = nn.elu(logit_scale) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale

    if query_mask is not None:
      if query_mask.ndim > 1:
        query_mask = jnp.expand_dims(query_mask, axis=-2)
      pred_logits = jnp.where(query_mask == 0, -1e6, pred_logits)

    return {'pred_logits': pred_logits, 'class_embeddings': image_class_emb}


class ImageTextEmbedder(nn.Module):
  """Embeds images and texts using selected backbone."""
  embed_configs: ml_collections.ConfigDict

  @nn.compact
  def __call__(
      self,
      *,
      images: Optional[jnp.ndarray] = None,
      texts: Optional[jnp.ndarray] = None,
      train: bool = False
  ) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Embeds text using selected backbone and configuration."""
    texts_shape = None
    if texts is not None:
      texts_shape = texts.shape
      if len(texts_shape) > 2:
        texts = texts.reshape(-1, texts_shape[-1])


    model_config = clip_model.CONFIGS[self.embed_configs['variant']]
    model_config['vision_return_map'] = True
    # Copy over additional CLIP config settings.
    for name in [
        'text_stochastic_droplayer_rate', 'vision_stochastic_droplayer_rate']:
      if self.embed_configs.get(name) is not None:
        model_config[name] = self.embed_configs[name]
    model = clip_layers.CLIP(**model_config, name='clip')

    # Input images should have range (0.0, 1.0). Shift them to CLIP range:
    if images is not None:
      images = clip_model.normalize_image(images)
    # Don't normalize image and text embeddings, similar to argus.
    img_emb, txt_emb = model(images, texts, normalize=False)

    # Drop or merge class embedding token.
    # TODO(mnn): Remove after the preferred class token merging scheme is
    # determined.
    if img_emb is not None:
      print("Image features", img_emb.shape)
      print(img_emb)
      merge_class_token = self.embed_configs.get('merge_class_token', 'sum')

      if merge_class_token == 'drop':
        img_emb = img_emb[:, 1:, :]   # [B, P, emb_dim]
      else:
        class_token_out = jnp.broadcast_to(
            img_emb[:, :1, :],
            np.array(img_emb.shape) - (0, 1, 0))
        if merge_class_token == 'sum':
          img_emb = img_emb[:, 1:, :] + class_token_out   # [B, P, emb_dim]
        elif merge_class_token == 'mul':
          img_emb = img_emb[:, 1:, :] * class_token_out   # [B, P, emb_dim]
        elif merge_class_token == 'sum-ln':
          img_emb = img_emb[:, 1:, :] + class_token_out   # [B, P, emb_dim]
          img_emb = nn.LayerNorm(name='merged_class_token')(img_emb)
        elif merge_class_token == 'mul-ln':
          img_emb = img_emb[:, 1:, :] * class_token_out   # [B, P, emb_dim]
          img_emb = nn.LayerNorm(name='merged_class_token')(img_emb)


    if txt_emb is not None and len(texts_shape) > 2:
      print("Text features", txt_emb.shape)
      print(txt_emb)
      txt_emb = txt_emb.reshape(texts_shape[:-1] + (-1,))
    return img_emb, txt_emb


class TextZeroShotDetectionModule(nn.Module):
  """Text-query-based ViT+ model with detection head.

  This module computes joint text and image embeddings which are then
  used for localized prediction of bboxes and classes.

  Attributes:
    body_configs: Configurations of the image-text module.
    normalize: Whether to normalize the output of the model and the
      label_embeddings before computing the class logits.
    box_bias: Type of box bias - one of 'location', 'size' or 'both'.
    mask_size: The height (and width) of masks predicted by the mask head. If
      None, no mask prediction will occur.
  """

  body_configs: ml_collections.ConfigDict
  normalize: bool = False
  box_bias: str = 'both'
  mask_size: Optional[int] = None

  @nn.nowrap
  def load_variables(self, checkpoint_path: str) -> Mapping[str, Any]:
    restored = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    return {'params': restored['optimizer']['target']}

  def setup(self):
    self._embedder = ImageTextEmbedder(self.body_configs, name='backbone')

    if 'out_dim' in self.body_configs:
      out_dim = self.body_configs.out_dim
    else:
      out_dim = clip_model.CONFIGS[self.body_configs.variant]['embed_dim']

    self._class_head = ClassPredictor(
        out_dim=out_dim,
        normalize=self.normalize, 
        name='class_head'
    )

    self._box_head = PredictorMLP(
        mlp_dim=None, 
        out_dim=4, 
        num_layers=3,
        out_activation=None, 
        name='obj_box_head'
    )

  def box_predictor(self, image_features: jnp.ndarray,
                    feature_map: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Computes predicted bounding boxes.

    Args:
      image_features: Features extracted from the image, returned by the
        `embedder` function.
      feature_map: A spatial re-arrangement of image_features, also returned by
        the `embedder` function.

    Returns:
      list of predicted boxes (cxcywh normalized to 0, 1) nested within
        a dictionary.
    """
    # Bounding box detection head [b, num_patches, 4].
    pred_boxes = self._box_head(image_features)
    # We compute the location of each token on the grid and use it to compute
    # a bias for the bbox prediction, i.e., each token is biased towards
    # predicting its location on the grid as the center.
    pred_boxes += utils.compute_box_bias(feature_map, kind=self.box_bias)
    pred_boxes = nn.sigmoid(pred_boxes)
    return {'pred_boxes': pred_boxes}

  def class_predictor(
      self,
      image_features: jnp.ndarray,
      query_embeddings: Optional[jnp.ndarray] = None,
      query_mask: Optional[jnp.ndarray] = None
  ) -> Dict[str, jnp.ndarray]:
  
    """Applies the class head to the image features.

    Args:
      image_features: Features extracted from the image embedder.
      query_embeddings: Optional list of (or image) embeddings. If no embeddings
        are provided, no logits will be computed and only the class embeddings
        for the image will be returned.
      query_mask: Must be provided with query_embeddings. A mask indicating
        which query embeddings are valid.

    Returns:
      A dictionary containing the class_embeddings and the pred_logits if
        query_embeddings and query_mask are provided.
    """
    return self._class_head(image_features, query_embeddings, query_mask)


  def image_embedder(self, images: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Embeds images into feature maps.

    Args:
      images: images of shape (batch, self.input_size, self.input_size, 3).
        Images should be in range [-1., 1.] with padding set to 0 and at the
        bottom right of the image.
      train: Whether or not we are in training mode.

    Returns:
      A 2D map of image features.
    """
    image_features, _ = self._embedder(images=images, train=train)
    return utils.seq2img(images, image_features)

  def text_embedder(self, text_queries: jnp.ndarray,
                    train: bool) -> jnp.ndarray:
    """Embeds text into features.

    Args:
      text_queries: jnp.int32 tokenized text queries of shape [..., num_tokens].
      train: Whether or not we are in training mode.

    Returns:
      An array of the same shape as text_queries, except for the last dimension,
      which is num_dimensions instead of num_tokens.
    """
    _, text_features = self._embedder(texts=text_queries, train=train)
    return text_features

  def __call__(self,
               inputs: jnp.ndarray,
               text_queries: jnp.ndarray,
               train: bool,
               *,
               debug: bool = False) -> Mapping[str, Any]:
    """Applies TextZeroShotDetectionModule on the input.

    Args:
      inputs: Images [batch_size, height, width, 3].
      text_queries: Queries to condition the model on. Queries starting with 0
        stand for padding [batch_size=b, num_queries=q, max_query_length=l].
      train: Whether it is training.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback. Not used.

    Returns:
      Outputs dict with items:
        pred_logits: Class logits [b, num_patches, num_queries + 1].
        pred_boxes: Predicted bounding boxes [b, num_patches, 4].
        feature_map: Image embeddings 2d feature map [b, sp, sp, img_emb_dim].
    """
    del debug
    # Embed images:
    feature_map = self.image_embedder(inputs, train)
    b, h, w, d = feature_map.shape
    image_features = jnp.reshape(feature_map, (b, h * w, d))

    # Embed queries:
    query_embeddings = self.text_embedder(text_queries, train)
    # If first token is 0, then this is a padded query [b, q].
    query_mask = (text_queries[..., 0] > 0).astype(jnp.float32)

    outputs = {
        'feature_map': feature_map,
        'query_embeddings': query_embeddings,
    }

    # Classification [b, num_patches, num_queries+1]:
    outputs.update(
        self.class_predictor(image_features, query_embeddings, query_mask))

    # Predict boxes:
    outputs.update(self.box_predictor(image_features, feature_map))

    return outputs
