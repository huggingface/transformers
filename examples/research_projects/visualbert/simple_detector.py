
# Reference: https://github.com/uclanlp/visualbert/blob/master/utils/detector.py
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet
import torch.nn.parallel
import torch.nn as nn
import torch


#from config_vcr import USE_IMAGENET_PRETRAINED

USE_IMAGENET_PRETRAINED = True


def pad_sequence(sequence, lengths):
    """
    :param sequence: [\sum b, .....] sequence
    :param lengths: [b1, b2, b3...] that sum to \sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output


class Flattener(torch.nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def _load_resnet(pretrained=True):
    # Reference: https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=False)
    if pretrained:
        backbone.load_state_dict(model_zoo.load_url(
            'https://s3.us-west-2.amazonaws.com/ai2-rowanz/resnet50-e13db6895d81.th'))
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    return backbone


def _load_resnet_imagenet(pretrained=True):
    # Reference: https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=pretrained)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone


class SimpleDetector(nn.Module):
    def __init__(self, pretrained=True, average_pool=True, semantic=True, final_dim=1024):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleDetector, self).__init__()
        # Reference: https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
        backbone = _load_resnet_imagenet(pretrained=pretrained) if USE_IMAGENET_PRETRAINED else _load_resnet(
            pretrained=pretrained)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            # backbone.layer4
        )

        from torchvision.layers import ROIAlign
        self.roi_align = ROIAlign((7, 7) if USE_IMAGENET_PRETRAINED else (14, 14),
                                  spatial_scale=1 / 16, sampling_ratio=0)

        if semantic:
            self.mask_dims = 32
            self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            self.mask_upsample = torch.nn.Conv2d(1, self.mask_dims, kernel_size=3,
                                                 stride=2 if USE_IMAGENET_PRETRAINED else 1,
                                                 padding=1, bias=True)
        else:
            self.object_embed = None
            self.mask_upsample = None

        after_roi_align = [backbone.layer4]
        self.final_dim = final_dim
        if average_pool:
            after_roi_align += [nn.AvgPool2d(7, stride=1), Flattener()]

        self.after_roi_align = torch.nn.Sequential(*after_roi_align)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048 + (128 if semantic else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.regularizing_predictor = torch.nn.Linear(2048, 81)

    def forward(self,
                images: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                classes: torch.Tensor = None,
                segms: torch.Tensor = None,
                ):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :return: object reps [batch_size, max_num_objects, dim]
        """
        # [batch_size, 2048, im_height // 32, im_width // 32
        img_feats = self.backbone(images)
        box_inds = box_mask.nonzero()
        assert box_inds.shape[0] > 0
        rois = torch.cat((
            box_inds[:, 0, None].type(boxes.dtype),
            boxes[box_inds[:, 0], box_inds[:, 1]],
        ), 1)
        # Object class and segmentation representations
        roi_align_res = self.roi_align(img_feats, rois)
        if self.mask_upsample is not None:
            assert segms is not None
            segms_indexed = segms[box_inds[:, 0], None, box_inds[:, 1]] - 0.5
            roi_align_res[:, :self.mask_dims] += self.mask_upsample(segms_indexed)
        post_roialign = self.after_roi_align(roi_align_res)

        # Add some regularization, encouraging the model to keep giving decent enough predictions
        obj_logits = self.regularizing_predictor(post_roialign)
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]
        cnn_regularization = F.cross_entropy(obj_logits, obj_labels, size_average=True)[None]

        feats_to_downsample = post_roialign if self.object_embed is None else torch.cat((post_roialign, self.object_embed(obj_labels)), -1)
        roi_aligned_feats = self.obj_downsample(feats_to_downsample)

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())
        return {
            'obj_reps_raw': post_roialign,
            'obj_reps': obj_reps,
            'obj_logits': obj_logits,
            'obj_labels': obj_labels,
            'cnn_regularization_loss': cnn_regularization
        }
