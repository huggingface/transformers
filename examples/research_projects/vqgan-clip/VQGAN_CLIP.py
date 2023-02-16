import os
from glob import glob

import imageio
import torch
import torchvision
import wandb
from img_processing import custom_to_pil, loop_post_process, preprocess, preprocess_vqgan
from loaders import load_vqgan
from PIL import Image
from torch import nn

from transformers import CLIPModel, CLIPTokenizerFast
from utils import get_device, get_timestamp, show_pil


class ProcessorGradientFlow:
    """
    This wraps the huggingface CLIP processor to allow backprop through the image processing step.
    The original processor forces conversion to PIL images, which is faster for image processing but breaks gradient flow.
    We call the original processor to get the text embeddings, but use our own image processing to keep images as torch tensors.
    """

    def __init__(self, device: str = "cpu", clip_model: str = "openai/clip-vit-large-patch14") -> None:
        self.device = device
        self.tokenizer = CLIPTokenizerFast.from_pretrained(clip_model)
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = torchvision.transforms.Normalize(self.image_mean, self.image_std)
        self.resize = torchvision.transforms.Resize(224)
        self.center_crop = torchvision.transforms.CenterCrop(224)

    def preprocess_img(self, images):
        images = self.resize(images)
        images = self.center_crop(images)
        images = self.normalize(images)
        return images

    def __call__(self, text=None, images=None, **kwargs):
        encoding = self.tokenizer(text=text, **kwargs)
        encoding["pixel_values"] = self.preprocess_img(images)
        encoding = {key: value.to(self.device) for (key, value) in encoding.items()}
        return encoding


class VQGAN_CLIP(nn.Module):
    def __init__(
        self,
        iterations=10,
        lr=0.01,
        vqgan=None,
        vqgan_config=None,
        vqgan_checkpoint=None,
        clip=None,
        clip_preprocessor=None,
        device=None,
        log=False,
        save_vector=True,
        return_val="image",
        quantize=True,
        save_intermediate=False,
        show_intermediate=False,
        make_grid=False,
    ) -> None:
        """
        Instantiate a VQGAN_CLIP model. If you want to use a custom VQGAN model, pass it as vqgan.
        """
        super().__init__()
        self.latent = None
        self.device = device if device else get_device()
        if vqgan:
            self.vqgan = vqgan
        else:
            self.vqgan = load_vqgan(self.device, conf_path=vqgan_config, ckpt_path=vqgan_checkpoint)
        self.vqgan.eval()
        if clip:
            self.clip = clip
        else:
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.to(self.device)
        self.clip_preprocessor = ProcessorGradientFlow(device=self.device)

        self.iterations = iterations
        self.lr = lr
        self.log = log
        self.make_grid = make_grid
        self.return_val = return_val
        self.quantize = quantize
        self.latent_dim = self.vqgan.decoder.z_shape

    def make_animation(self, input_path=None, output_path=None, total_duration=5, extend_frames=True):
        """
        Make an animation from the intermediate images saved during generation.
        By default, uses the images from the most recent generation created by the generate function.
        If you want to use images from a different generation, pass the path to the folder containing the images as input_path.
        """
        images = []
        if output_path is None:
            output_path = "./animation.gif"
        if input_path is None:
            input_path = self.save_path
        paths = list(sorted(glob(input_path + "/*")))
        if not len(paths):
            raise ValueError(
                "No images found in save path, aborting (did you pass save_intermediate=True to the generate"
                " function?)"
            )
        if len(paths) == 1:
            print("Only one image found in save path, (did you pass save_intermediate=True to the generate function?)")
        frame_duration = total_duration / len(paths)
        durations = [frame_duration] * len(paths)
        if extend_frames:
            durations[0] = 1.5
            durations[-1] = 3
        for file_name in paths:
            if file_name.endswith(".png"):
                images.append(imageio.imread(file_name))
        imageio.mimsave(output_path, images, duration=durations)
        print(f"gif saved to {output_path}")

    def _get_latent(self, path=None, img=None):
        if not (path or img):
            raise ValueError("Input either path or tensor")
        if img is not None:
            raise NotImplementedError
        x = preprocess(Image.open(path), target_image_size=256).to(self.device)
        x_processed = preprocess_vqgan(x)
        z, *_ = self.vqgan.encode(x_processed)
        return z

    def _add_vector(self, transform_vector):
        """Add a vector transform to the base latent and returns the resulting image."""
        base_latent = self.latent.detach().requires_grad_()
        trans_latent = base_latent + transform_vector
        if self.quantize:
            z_q, *_ = self.vqgan.quantize(trans_latent)
        else:
            z_q = trans_latent
        return self.vqgan.decode(z_q)

    def _get_clip_similarity(self, prompts, image, weights=None):
        clip_inputs = self.clip_preprocessor(text=prompts, images=image, return_tensors="pt", padding=True)
        clip_outputs = self.clip(**clip_inputs)
        similarity_logits = clip_outputs.logits_per_image
        if weights is not None:
            similarity_logits = similarity_logits * weights
        return similarity_logits.sum()

    def _get_clip_loss(self, pos_prompts, neg_prompts, image):
        pos_logits = self._get_clip_similarity(pos_prompts["prompts"], image, weights=(1 / pos_prompts["weights"]))
        if neg_prompts:
            neg_logits = self._get_clip_similarity(neg_prompts["prompts"], image, weights=neg_prompts["weights"])
        else:
            neg_logits = torch.tensor([1], device=self.device)
        loss = -torch.log(pos_logits) + torch.log(neg_logits)
        return loss

    def _optimize_CLIP(self, original_img, pos_prompts, neg_prompts):
        vector = torch.randn_like(self.latent, requires_grad=True, device=self.device)
        optim = torch.optim.Adam([vector], lr=self.lr)

        for i in range(self.iterations):
            optim.zero_grad()
            transformed_img = self._add_vector(vector)
            processed_img = loop_post_process(transformed_img)
            clip_loss = self._get_CLIP_loss(pos_prompts, neg_prompts, processed_img)
            print("CLIP loss", clip_loss)
            if self.log:
                wandb.log({"CLIP Loss": clip_loss})
            clip_loss.backward(retain_graph=True)
            optim.step()
            if self.return_val == "image":
                yield custom_to_pil(transformed_img[0])
            else:
                yield vector

    def _init_logging(self, positive_prompts, negative_prompts, image_path):
        wandb.init(reinit=True, project="face-editor")
        wandb.config.update({"Positive Prompts": positive_prompts})
        wandb.config.update({"Negative Prompts": negative_prompts})
        wandb.config.update(dict(lr=self.lr, iterations=self.iterations))
        if image_path:
            image = Image.open(image_path)
            image = image.resize((256, 256))
            wandb.log("Original Image", wandb.Image(image))

    def process_prompts(self, prompts):
        if not prompts:
            return []
        processed_prompts = []
        weights = []
        if isinstance(prompts, str):
            prompts = [prompt.strip() for prompt in prompts.split("|")]
        for prompt in prompts:
            if isinstance(prompt, (tuple, list)):
                processed_prompt = prompt[0]
                weight = float(prompt[1])
            elif ":" in prompt:
                processed_prompt, weight = prompt.split(":")
                weight = float(weight)
            else:
                processed_prompt = prompt
                weight = 1.0
            processed_prompts.append(processed_prompt)
            weights.append(weight)
        return {
            "prompts": processed_prompts,
            "weights": torch.tensor(weights, device=self.device),
        }

    def generate(
        self,
        pos_prompts,
        neg_prompts=None,
        image_path=None,
        show_intermediate=True,
        save_intermediate=False,
        show_final=True,
        save_final=True,
        save_path=None,
    ):
        """Generate an image from the given prompts.
        If image_path is provided, the image is used as a starting point for the optimization.
        If image_path is not provided, a random latent vector is used as a starting point.
        You must provide at least one positive prompt, and optionally provide negative prompts.
        Prompts must be formatted in one of the following ways:
        - A single prompt as a string, e.g "A smiling woman"
        - A set of prompts separated by pipes: "A smiling woman | a woman with brown hair"
        - A set of prompts and their weights separated by colons: "A smiling woman:1 | a woman with brown hair: 3" (default weight is 1)
        - A list of prompts, e.g ["A smiling woman", "a woman with brown hair"]
        - A list of prompts and weights, e.g [("A smiling woman", 1), ("a woman with brown hair", 3)]
        """
        if image_path:
            self.latent = self._get_latent(image_path)
        else:
            self.latent = torch.randn(self.latent_dim, device=self.device)
        if self.log:
            self._init_logging(pos_prompts, neg_prompts, image_path)

        assert pos_prompts, "You must provide at least one positive prompt."
        pos_prompts = self.process_prompts(pos_prompts)
        neg_prompts = self.process_prompts(neg_prompts)
        if save_final and save_path is None:
            save_path = os.path.join("./outputs/", "_".join(pos_prompts["prompts"]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            save_path = save_path + "_" + get_timestamp()
            os.makedirs(save_path)
        self.save_path = save_path

        original_img = self.vqgan.decode(self.latent)[0]
        if show_intermediate:
            print("Original Image")
            show_pil(custom_to_pil(original_img))

        original_img = loop_post_process(original_img)
        for iter, transformed_img in enumerate(self._optimize_CLIP(original_img, pos_prompts, neg_prompts)):
            if show_intermediate:
                show_pil(transformed_img)
            if save_intermediate:
                transformed_img.save(os.path.join(self.save_path, f"iter_{iter:03d}.png"))
            if self.log:
                wandb.log({"Image": wandb.Image(transformed_img)})
        if show_final:
            show_pil(transformed_img)
        if save_final:
            transformed_img.save(os.path.join(self.save_path, f"iter_{iter:03d}_final.png"))
