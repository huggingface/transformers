from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import FastConfig, PreTrainedModel
from transformers.utils import ModelOutput


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class My2DLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, use_bn=True, act_func="relu", dropout_rate=0, ops_order="weight_bn_act"
    ):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules"""
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm2d(in_channels)
            else:
                modules["bn"] = nn.BatchNorm2d(out_channels)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule"""

    def forward(self, x):
        for key, module in self._modules.items():
            if key == "bn" and not self.training:
                continue
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        use_act=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.act_func = act_func

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.Identity()
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.Identity()
        if use_act:
            act = build_activation(self.act_func, True)
            if act is not None:
                self.act = act

    def forward(self, x):
        if self.training:
            if hasattr(self, "fused_conv"):
                delattr(self, "fused_conv")
            x = self.conv(x)
            x = self.bn(x)
            return self.act(x)
        else:
            if not hasattr(self, "fused_conv"):
                setattr(self, "fused_conv", self.fuse_conv_bn(self.conv, self.bn))
            x = self.fused_conv(x)
            if self.act is not None:
                x = self.act(x)
            return x

    def fuse_conv_bn(self, conv, bn):
        """During inference, the functionary of batch norm layers is turned off but
        only the mean and var alone channels are used, which exposes the chance to fuse it with the preceding conv
        layers to save computations and simplify network structures."""
        if isinstance(bn, nn.Identity):
            return conv
        conv_w = conv.weight
        conv_b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
        conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
        return conv


class RepConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super(RepConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        assert len(kernel_size) == 2
        padding = (int(((kernel_size[0] - 1) * dilation) / 2), int(((kernel_size[1] - 1) * dilation) / 2))

        self.nonlinearity = nn.ReLU(inplace=True)

        self.main_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.main_bn = nn.BatchNorm2d(num_features=out_channels)

        ver_pad = (int(((kernel_size[0] - 1) * dilation) / 2), 0)
        hor_pad = (0, int(((kernel_size[1] - 1) * dilation) / 2))

        if kernel_size[1] != 1:
            self.ver_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=stride,
                padding=ver_pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.ver_conv, self.ver_bn = None, None

        if kernel_size[0] != 1:  # 卷积核的高大于1 -> 有水平卷积
            self.hor_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size[1]),
                stride=stride,
                padding=hor_pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.hor_conv, self.hor_bn = None, None

        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        )

    def forward(self, input):
        if self.training:
            if hasattr(self, "fused_conv"):
                self.__delattr__("fused_conv")

            main_outputs = self.main_conv(input)
            main_outputs = self.main_bn(main_outputs)
            if self.ver_conv is not None:
                vertical_outputs = self.ver_conv(input)
                vertical_outputs = self.ver_bn(vertical_outputs)
            else:
                vertical_outputs = 0

            if self.hor_conv is not None:
                horizontal_outputs = self.hor_conv(input)
                horizontal_outputs = self.hor_bn(horizontal_outputs)
            else:
                horizontal_outputs = 0

            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(input)

            return self.nonlinearity(main_outputs + vertical_outputs + horizontal_outputs + id_out)
        else:
            if not hasattr(self, "fused_conv"):
                self.prepare_for_eval()
            return self.nonlinearity(self.fused_conv(input))

    def _identity_to_conv(self, identity):
        if identity is None:
            return 0, 0
        assert isinstance(identity, nn.BatchNorm2d)
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        running_mean = identity.running_mean
        running_var = identity.running_var
        gamma = identity.weight
        beta = identity.bias
        eps = identity.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel_mxn, bias_mxn = self._fuse_bn_tensor(self.main_conv, self.main_bn)
        if self.ver_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        else:
            kernel_mx1, bias_mx1 = 0, 0
        if self.hor_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        else:
            kernel_1xn, bias_1xn = 0, 0
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel):
        kernel_height, kernel_width = self.kernel_size
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right, pad_top_down, pad_top_down])

    def prepare_for_eval(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.main_conv.in_channels,
            out_channels=self.main_conv.out_channels,
            kernel_size=self.main_conv.kernel_size,
            stride=self.main_conv.stride,
            padding=self.main_conv.padding,
            dilation=self.main_conv.dilation,
            groups=self.main_conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.fused_conv.parameters():
            para.detach_()


class FastPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastConfig
    base_model_prefix = "fast"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


class TextNet(FastPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.first_conv = ConvLayer(
            config.backbone_in_channels,
            config.backbone_out_channels,
            config.backbone_kernel_size,
            config.backbone_stride,
            config.backbone_dilation,
            config.backbone_groups,
            config.backbone_bias,
            config.backbone_has_shuffle,
            config.backbone_use_bn,
            config.backbone_act_func,
            config.backbone_dropout_rate,
            config.backbone_ops_order,
        )
        self.first_conv.apply(self._init_weights)
        stage1 = []
        for stage_config in zip(
            config.backbone_stage1_in_channels,
            config.backbone_stage1_out_channels,
            config.backbone_stage1_kernel_size,
            config.backbone_stage1_stride,
            config.backbone_stage1_dilation,
            config.backbone_stage1_groups,
        ):
            stage1.append(RepConvLayer(*stage_config))
        self.stage1 = nn.ModuleList(stage1)

        stage2 = []
        for stage_config in zip(
            config.backbone_stage2_in_channels,
            config.backbone_stage2_out_channels,
            config.backbone_stage2_kernel_size,
            config.backbone_stage2_stride,
            config.backbone_stage2_dilation,
            config.backbone_stage2_groups,
        ):
            stage2.append(RepConvLayer(*stage_config))
        self.stage2 = nn.ModuleList(stage2)

        stage3 = []
        for stage_config in zip(
            config.backbone_stage3_in_channels,
            config.backbone_stage3_out_channels,
            config.backbone_stage3_kernel_size,
            config.backbone_stage3_stride,
            config.backbone_stage3_dilation,
            config.backbone_stage3_groups,
        ):
            stage3.append(RepConvLayer(*stage_config))
        self.stage3 = nn.ModuleList(stage3)

        stage4 = []
        for stage_config in zip(
            config.backbone_stage4_in_channels,
            config.backbone_stage4_out_channels,
            config.backbone_stage4_kernel_size,
            config.backbone_stage4_stride,
            config.backbone_stage4_dilation,
            config.backbone_stage4_groups,
        ):
            stage4.append(RepConvLayer(*stage_config))
        self.stage4 = nn.ModuleList(stage4)

    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.first_conv(x)
        output = []

        for block in self.stage1:
            x = block(x)
        output.append(x)

        for block in self.stage2:
            x = block(x)
        output.append(x)

        for block in self.stage3:
            x = block(x)
        output.append(x)

        for block in self.stage4:
            x = block(x)
        output.append(x)

        return output


class FASTNeck(FastPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        reduce_layer_configs = list(
            zip(
                config.neck_in_channels,
                config.neck_out_channels,
                config.neck_kernel_size,
                config.neck_stride,
                config.neck_dilation,
                config.neck_groups,
            )
        )
        self.layers_count = len(reduce_layer_configs)
        for layer_ix in range(0, len(reduce_layer_configs)):
            setattr(self, f"reduce_layer{layer_ix + 1}", RepConvLayer(*reduce_layer_configs[layer_ix]))
        # self.reduce_layer1 = RepConvLayer(*reduce_layer_configs[0])
        # self.reduce_layer2 = RepConvLayer(*reduce_layer_configs[1])
        # self.reduce_layer3 = RepConvLayer(*reduce_layer_configs[2])
        # self.reduce_layer4 = RepConvLayer(*reduce_layer_configs[3])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear")

    def forward(self, x):
        f1 = x[0]
        f1 = self.reduce_layer1(f1)
        output_stages = [f1]

        for layer_ix in range(1, self.layers_count):
            layer_out = getattr(self, f"reduce_layer{layer_ix + 1}")(x[layer_ix])
            layer_out = self._upsample(layer_out, f1)
            output_stages.append(layer_out)

        f = torch.cat(output_stages, 1)
        return f


class FASTHead(FastPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.conv = RepConvLayer(
            config.head_conv_in_channels,
            config.head_conv_out_channels,
            config.head_conv_kernel_size,
            config.head_conv_stride,
            config.head_conv_dilation,
            config.head_conv_groups,
        )

        self.final = ConvLayer(
            config.head_final_in_channels,
            config.head_final_out_channels,
            config.head_final_kernel_size,
            config.head_final_stride,
            config.head_final_dilation,
            config.head_final_groups,
            config.head_final_bias,
            config.head_final_has_shuffle,
            config.head_final_use_bn,
            config.head_final_act_func,
            config.head_final_dropout_rate,
            config.head_final_ops_order,
        )

        self.min_area = config.min_area
        self.min_score = config.min_score
        self.bbox_type = config.bbox_type

        self.pooling_size = config.head_pooling_size

        self.pooling_1s = nn.MaxPool2d(kernel_size=self.pooling_size, stride=1, padding=(self.pooling_size - 1) // 2)
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=self.pooling_size // 2 + 1, stride=1, padding=(self.pooling_size // 2) // 2
        )

        if config.head_dropout_ratio > 0:
            self.dropout = nn.Dropout2d(config.head_dropout_ratio)
        else:
            self.dropout = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.final(x)
        return x

    # def get_results(self, out, img_meta, scale=2):
    #     org_img_size = img_meta["org_img_size"]
    #     img_size = img_meta["img_size"]  # 640*640
    #     batch_size = out.size(0)
    #     outputs = {}
    #
    #     texts = F.interpolate(
    #         out[:, 0:1, :, :], size=(img_size[0] // scale, img_size[1] // scale), mode="nearest"
    #     )  # B*1*320*320
    #     texts = self._max_pooling(texts, scale=scale)  # B*1*320*320
    #     score_maps = torch.sigmoid_(texts)  # B*1*320*320~
    #     score_maps = F.interpolate(score_maps, size=(img_size[0], img_size[1]), mode="nearest")  # B*1*640*640
    #     score_maps = score_maps.squeeze(1)  # B*640*640
    #
    #     kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B*160*160
    #     labels_ = []
    #     for kernel in kernels.numpy():
    #         ret, label_ = cv2.connectedComponents(kernel)
    #         labels_.append(label_)
    #     labels_ = np.array(labels_)
    #     labels_ = torch.from_numpy(labels_)
    #     labels = labels_.unsqueeze(1).to(torch.float32)  # B*1*160*160
    #     labels = F.interpolate(
    #         labels, size=(img_size[0] // scale, img_size[1] // scale), mode="nearest"
    #     )  # B*1*320*320
    #     labels = self._max_pooling(labels, scale=scale)
    #     labels = F.interpolate(labels, size=(img_size[0], img_size[1]), mode="nearest")  # B*1*640*640
    #     labels = labels.squeeze(1).to(torch.int32)  # B*640*640
    #
    #     keys = [torch.unique(labels_[i], sorted=True) for i in range(batch_size)]
    #
    #     outputs.update({"kernels": kernels.data.cpu()})
    #
    #     scales = (float(org_img_size[1]) / float(img_size[1]), float(org_img_size[0]) / float(img_size[0]))
    #
    #     results = []
    #     for i in range(batch_size):
    #         bboxes, scores = self.generate_bbox(keys[i], labels[i], score_maps[i], scales)
    #         results.append({"bboxes": bboxes, "scores": scores})
    #     outputs.update({"results": results})
    #
    #     return outputs

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x

    # def generate_bbox(self, keys, label, score, scales):
    #     label_num = len(keys)
    #     bboxes = []
    #     scores = []
    #     for index in range(1, label_num):
    #         i = keys[index]
    #         ind = label == i
    #         ind_np = ind.data.cpu().numpy()
    #         points = np.array(np.where(ind_np)).transpose((1, 0))
    #         if points.shape[0] < self.min_area:
    #             label[ind] = 0
    #             continue
    #         score_i = score[ind].mean().item()
    #         if score_i < self.min_score:
    #             label[ind] = 0
    #             continue
    #
    #         if self.bbox_type == "rect":
    #             rect = cv2.minAreaRect(points[:, ::-1])
    #             alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
    #             rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
    #             bbox = cv2.boxPoints(rect) * scales
    #         else:
    #             binary = np.zeros(label.shape, dtype="uint8")
    #             binary[ind_np] = 1
    #             contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             bbox = contours[0] * scales
    #
    #         bbox = bbox.astype("int32")
    #         bboxes.append(bbox.reshape(-1).tolist())
    #         scores.append(score_i)
    #     return bboxes, scores


def emb_loss(
    emb, instance, kernel, training_mask, feature_dim=4, delta_v=0.5, delta_d=1.5, weights=(1.0, 1.0), bg_sample=False
):
    training_mask = (training_mask > 0.5).long()
    kernel = (kernel > 0.5).long()
    instance = instance * training_mask
    instance_kernel = (instance * kernel).view(-1)
    instance = instance.view(-1)
    emb = emb.view(feature_dim, -1)

    unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
    num_instance = unique_labels.size(0)
    if num_instance <= 1:
        return 0

    emb_mean = emb.new_zeros((feature_dim, num_instance), dtype=torch.float32)
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind_k = instance_kernel == lb
        emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

    l_agg = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind = instance == lb
        emb_ = emb[:, ind]
        dist = (emb_ - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
        dist = F.relu(dist - delta_v) ** 2
        l_agg[i] = torch.mean(torch.log(dist + 1.0))
    l_agg = torch.mean(l_agg[1:])

    if num_instance > 2:
        emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
        emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, feature_dim)
        # print(seg_band)

        mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, feature_dim)
        mask = mask.view(num_instance, num_instance, -1)
        mask[0, :, :] = 0
        mask[:, 0, :] = 0
        mask = mask.view(num_instance * num_instance, -1)
        # print(mask)

        dist = emb_interleave - emb_band
        dist = dist[mask > 0].view(-1, feature_dim).norm(p=2, dim=1)
        dist = F.relu(2 * delta_d - dist) ** 2
        l_dis = torch.mean(torch.log(dist + 1.0))

        if bg_sample:
            l_dis = [torch.log(dist + 1.0)]
            emb_bg = emb[:, instance == 0].view(feature_dim, -1)
            if emb_bg.size(1) > 100:
                rand_ind = np.random.permutation(emb_bg.size(1))[:100]
                emb_bg = emb_bg[:, rand_ind]
            if emb_bg.size(1) > 0:
                for i, lb in enumerate(unique_labels):
                    if lb == 0:
                        continue
                    dist = (emb_bg - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
                    dist = F.relu(2 * delta_d - dist) ** 2
                    l_dis_bg = torch.mean(torch.log(dist + 1.0), 0, keepdim=True)
                    l_dis.append(l_dis_bg)
            l_dis = torch.mean(torch.cat(l_dis))
    else:
        l_dis = 0

    l_agg = weights[0] * l_agg
    l_dis = weights[1] * l_dis
    l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
    loss = l_agg + l_dis + l_reg
    return loss


def emb_loss_batch(emb, instance, kernel, training_mask, reduce=True, loss_weight=0.25, bg_sample=False):
    loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

    for i in range(loss_batch.size(0)):
        loss_batch[i] = emb_loss(emb[i], instance[i], kernel[i], training_mask[i])

    loss_batch = loss_weight * loss_batch

    if reduce:
        loss_batch = torch.mean(loss_batch)

    return loss_batch


def dice_loss_with_masks(input, target, mask, reduce=True):
    loss_weight = 0.5
    batch_size = input.size(0)
    input = torch.sigmoid(input)

    input = input.contiguous().view(batch_size, -1)
    target = target.contiguous().view(batch_size, -1).float()
    mask = mask.contiguous().view(batch_size, -1).float()

    input = input * mask
    target = target * mask

    a = torch.sum(input * target, dim=1)
    b = torch.sum(input * input, dim=1) + 0.001
    c = torch.sum(target * target, dim=1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d

    loss = loss_weight * loss

    if reduce:
        loss = torch.mean(loss)

    return loss


def ohem_single(score, gt_text, training_mask):
    pos_num = int(torch.sum(gt_text > 0.5)) - int(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_num = int(torch.sum(gt_text <= 0.5))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted, _ = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks


def iou_single(a, b, mask, n_class):
    EPS = 1e-6
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []
    for i in range(n_class):
        inter = ((a == i) & (b == i)).float()
        union = ((a == i) | (b == i)).float()

        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.size(0)

    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    iou = a.new_zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = torch.mean(iou)
    return iou


@dataclass
class FASTForImageCaptioningOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        text_hidden (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional*):
            The image hidden states.
    """

    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.FloatTensor] = None


class FASTForImageCaptioning(FastPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = TextNet(config=config)
        self.neck = FASTNeck(config=config)
        self.det_head = FASTHead(config=config)
        self.loss_bg = config.loss_bg

        self.pooling_1s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size, stride=1, padding=(config.head_pooling_size - 1) // 2
        )
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size // 2 + 1, stride=1, padding=(config.head_pooling_size // 2) // 2
        )
        self.post_init()

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode="bilinear")

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x

    def loss(self, hidden, labels):
        gt_texts = labels["gt_texts"]
        gt_kernels = labels["gt_kernels"]
        training_masks = labels["training_masks"]
        gt_instances = labels["gt_instances"]

        kernels = hidden[:, 0, :, :]  # 4*640*640
        texts = self._max_pooling(kernels, scale=1)  # 4*640*640
        embs = hidden[:, 1:, :, :]  # 4*4*640*640

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = dice_loss_with_masks(texts, gt_texts, selected_masks, reduce=False)

        selected_masks = gt_texts * training_masks
        loss_kernel = dice_loss_with_masks(kernels, gt_kernels, selected_masks, reduce=False)
        loss_kernel = torch.mean(loss_kernel, dim=0)

        loss_emb = emb_loss_batch(embs, gt_instances, gt_kernels, training_masks, reduce=False, bg_sample=self.loss_bg)

        return torch.mean(loss_text) + torch.mean(loss_kernel) + torch.mean(loss_emb)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        labels: Dict = None,
    ):
        # outputs = {}
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        f = self.backbone(pixel_values)

        f = self.neck(f)

        det_out = self.det_head(f)

        loss = None
        if labels:
            out = self._upsample(det_out, pixel_values.size(), scale=1)
            loss = self.loss(out, labels)
        # det_res = self.det_head.get_results(det_out, img_metas, scale=2)
        # outputs.update(det_res)
        det_out = self._upsample(det_out, pixel_values.size(), scale=4)

        if not return_dict:
            return (loss, det_out) if loss is not None else (det_out,)

        return FASTForImageCaptioningOutput(loss, det_out)
