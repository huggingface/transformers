import math
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel


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
        for module in self._modules.values():
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


class FalsePreTrainedModel(PreTrainedModel):
    pass


class ConvLayer(My2DLayer):
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
            ops_order="weight_bn_act",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict["conv"] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

        return weight_dict


class RepConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, deploy=False):
        super(RepConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy

        assert len(kernel_size) == 2
        padding = (int(((kernel_size[0] - 1) * dilation) / 2), int(((kernel_size[1] - 1) * dilation) / 2))

        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
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

            if kernel_size[1] != 1:  # 卷积核的宽大于1 -> 有垂直卷积
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
        if hasattr(self, "fused_conv"):
            return self.nonlinearity(self.fused_conv(input))
        else:
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

    # def switch_to_deploy(self):
    #     if hasattr(self, 'fused_conv'):
    #         return
    #     kernel, bias = self.get_equivalent_kernel_bias()
    #     self.fused_conv = nn.Conv2d(in_channels=self.main_conv.in_channels,
    #                                 out_channels=self.main_conv.out_channels,
    #                                 kernel_size=self.main_conv.kernel_size, stride=self.main_conv.stride,
    #                                 padding=self.main_conv.padding, dilation=self.main_conv.dilation,
    #                                 groups=self.main_conv.groups, bias=True)
    #     self.fused_conv.weight.data = kernel
    #     self.fused_conv.bias.data = bias
    #     self.deploy = True
    #     for para in self.parameters():
    #         para.detach_()
    #     for attr in ['main_conv', 'main_bn', 'ver_conv', 'ver_bn', 'hor_conv', 'hor_bn']:
    #         if hasattr(self, attr):
    #             self.__delattr__(attr)
    #
    #     if hasattr(self, 'rbr_identity'):
    #         self.__delattr__('rbr_identity')

    # def switch_to_test(self):
    #     kernel, bias = self.get_equivalent_kernel_bias()
    #     self.fused_conv = nn.Conv2d(in_channels=self.main_conv.in_channels,
    #                                 out_channels=self.main_conv.out_channels,
    #                                 kernel_size=self.main_conv.kernel_size, stride=self.main_conv.stride,
    #                                 padding=self.main_conv.padding, dilation=self.main_conv.dilation,
    #                                 groups=self.main_conv.groups, bias=True)
    #     self.fused_conv.weight.data = kernel
    #     self.fused_conv.bias.data = bias
    #     for para in self.fused_conv.parameters():
    #         para.detach_()
    #     self.deploy = True

    # def switch_to_train(self):
    #     if hasattr(self, 'fused_conv'):
    #         self.__delattr__('fused_conv')
    #     self.deploy = False

    # @staticmethod
    # def is_zero_layer():
    #     return False

    # @property
    # def module_str(self):
    #     return 'Rep_%dx%d' % (self.kernel_size[0], self.kernel_size[1])

    # @property
    # def config(self):
    #     return {'name': RepConvLayer.__name__,
    #             'in_channels': self.in_channels,
    #             'out_channels': self.out_channels,
    #             'kernel_size': self.kernel_size,
    #             'stride': self.stride,
    #             'dilation': self.dilation,
    #             'groups': self.groups}

    # @staticmethod
    # def build_from_config(config):
    #     return RepConvLayer(**config)


class TextNet(PreTrainedModel):
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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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


class FASTNeck(PreTrainedModel):
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

        self.reduce_layer1 = RepConvLayer(*reduce_layer_configs[0])
        self.reduce_layer2 = RepConvLayer(*reduce_layer_configs[1])
        self.reduce_layer3 = RepConvLayer(*reduce_layer_configs[2])
        self.reduce_layer4 = RepConvLayer(*reduce_layer_configs[3])

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
        f1, f2, f3, f4 = x
        f1 = self.reduce_layer1(f1)
        f2 = self.reduce_layer2(f2)
        f3 = self.reduce_layer3(f3)
        f4 = self.reduce_layer4(f4)

        f2 = self._upsample(f2, f1)
        f3 = self._upsample(f3, f1)
        f4 = self._upsample(f4, f1)
        f = torch.cat((f1, f2, f3, f4), 1)
        return f


class FASTHead(nn.Module):
    def __init__(self, config):
        super(FASTHead, self).__init__()
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


class FASTForImageCaptioning(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = TextNet(config=config)
        self.neck = FASTNeck(config=config)
        self.det_head = FASTHead(config=config)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode="bilinear")

    def forward(self, imgs, img_metas=None):
        outputs = {}

        f = self.backbone(imgs)

        f = self.neck(f)

        det_out = self.det_head(f)

        det_out = self._upsample(det_out, imgs.size(), scale=4)
        # det_res = self.det_head.get_results(det_out, img_metas, scale=2)
        # outputs.update(det_res)

        return det_out
