import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2

class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
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

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss


class EmbLoss_v1(nn.Module):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.0, 1.0)

    def forward_single(self, emb, instance, kernel, training_mask):
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)

        unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((self.feature_dim, num_instance), dtype=torch.float32)
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
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - self.delta_v) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, self.feature_dim)
            # print(seg_band)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)
            # print(mask)

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_d - dist) ** 2
            l_dis = torch.mean(torch.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb, instance, kernel, training_mask, reduce=True):
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i], training_mask[i])

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = torch.mean(loss_batch)

        return loss_batch


class EmbLoss_v2(nn.Module):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v2, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.0, 1.0)

    def forward_single(self, emb, instance, kernel, training_mask):
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)

        unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((self.feature_dim, num_instance), dtype=torch.float32)
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
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - self.delta_v) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, self.feature_dim)
            # print(seg_band)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)
            # print(mask)

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_d - dist) ** 2
            # l_dis = torch.mean(torch.log(dist + 1.0))

            l_dis = [torch.log(dist + 1.0)]
            emb_bg = emb[:, instance == 0].view(self.feature_dim, -1)
            if emb_bg.size(1) > 100:
                rand_ind = np.random.permutation(emb_bg.size(1))[:100]
                emb_bg = emb_bg[:, rand_ind]
            if emb_bg.size(1) > 0:
                for i, lb in enumerate(unique_labels):
                    if lb == 0:
                        continue
                    dist = (emb_bg - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
                    dist = F.relu(2 * self.delta_d - dist) ** 2
                    l_dis_bg = torch.mean(torch.log(dist + 1.0), 0, keepdim=True)
                    l_dis.append(l_dis_bg)
            l_dis = torch.mean(torch.cat(l_dis))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb, instance, kernel, training_mask, reduce=True):
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i], training_mask[i])

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = torch.mean(loss_batch)

        return loss_batch


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        RepConvLayer.__name__: RepConvLayer
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class My2DLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

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
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


def generate_bbox(keys, label, score, scales, cfg):
    label_num = len(keys)
    bboxes = []
    scores = []
    for index in range(1, label_num):
        i = keys[index]
        ind = (label == i)
        ind_np = ind.data.cpu().numpy()
        points = np.array(np.where(ind_np)).transpose((1, 0))
        if points.shape[0] < cfg.test_cfg.min_area:
            label[ind] = 0
            continue
        score_i = score[ind].mean().item()
        if score_i < cfg.test_cfg.min_score:
            label[ind] = 0
            continue

        if cfg.test_cfg.bbox_type == 'rect':
            rect = cv2.minAreaRect(points[:, ::-1])
            alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
            rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
            bbox = cv2.boxPoints(rect) * scales

        elif cfg.test_cfg.bbox_type == 'poly':
            binary = np.zeros(label.shape, dtype='uint8')
            binary[ind_np] = 1
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox = contours[0] * scales
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1).tolist())
        scores.append(score_i)
    return bboxes, scores


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
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
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )

        return weight_dict

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


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
        padding = (int(((kernel_size[0] - 1) * dilation) / 2),
                   int(((kernel_size[1] - 1) * dilation) / 2))

        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=True)
        else:
            self.main_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                       dilation=dilation, groups=groups, bias=False)
            self.main_bn = nn.BatchNorm2d(num_features=out_channels)

            ver_pad = (int(((kernel_size[0] - 1) * dilation) / 2), 0)
            hor_pad = (0, int(((kernel_size[1] - 1) * dilation) / 2))

            if kernel_size[1] != 1:  # 卷积核的宽大于1 -> 有垂直卷积
                self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=(kernel_size[0], 1),
                                          stride=stride, padding=ver_pad,
                                          dilation=dilation, groups=groups, bias=False)
                self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.ver_conv, self.ver_bn = None, None

            if kernel_size[0] != 1:  # 卷积核的高大于1 -> 有水平卷积
                self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=(1, kernel_size[1]),
                                          stride=stride, padding=hor_pad,
                                          dilation=dilation, groups=groups, bias=False)
                self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.hor_conv, self.hor_bn = None, None

            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, input):
        if hasattr(self, 'fused_conv'):
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
        if not hasattr(self, 'id_tensor'):
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
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right,
                                                pad_top_down, pad_top_down])

    def switch_to_deploy(self):
        if hasattr(self, 'fused_conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(in_channels=self.main_conv.in_channels,
                                    out_channels=self.main_conv.out_channels,
                                    kernel_size=self.main_conv.kernel_size, stride=self.main_conv.stride,
                                    padding=self.main_conv.padding, dilation=self.main_conv.dilation,
                                    groups=self.main_conv.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        self.deploy = True
        for para in self.parameters():
            para.detach_()
        for attr in ['main_conv', 'main_bn', 'ver_conv', 'ver_bn', 'hor_conv', 'hor_bn']:
            if hasattr(self, attr):
                self.__delattr__(attr)

        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

    def switch_to_test(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(in_channels=self.main_conv.in_channels,
                                    out_channels=self.main_conv.out_channels,
                                    kernel_size=self.main_conv.kernel_size, stride=self.main_conv.stride,
                                    padding=self.main_conv.padding, dilation=self.main_conv.dilation,
                                    groups=self.main_conv.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.fused_conv.parameters():
            para.detach_()
        self.deploy = True

    def switch_to_train(self):
        if hasattr(self, 'fused_conv'):
            self.__delattr__('fused_conv')
        self.deploy = False

    @staticmethod
    def is_zero_layer():
        return False

    @property
    def module_str(self):
        return 'Rep_%dx%d' % (self.kernel_size[0], self.kernel_size[1])

    @property
    def config(self):
        return {'name': RepConvLayer.__name__,
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'dilation': self.dilation,
                'groups': self.groups}

    @staticmethod
    def build_from_config(config):
        return RepConvLayer(**config)


class TextNet(nn.Module):

    def __init__(self, first_conv, stage1, stage2, stage3, stage4):
        super(TextNet, self).__init__()

        self.first_conv = first_conv
        self.stage1 = nn.ModuleList(stage1)
        self.stage2 = nn.ModuleList(stage2)
        self.stage3 = nn.ModuleList(stage3)
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
        output = list()

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

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        stage1, stage2, stage3, stage4 = [], [], [], []
        for block_config in config['stage1']:
            stage1.append(set_layer_from_config(block_config))
        for block_config in config['stage2']:
            stage2.append(set_layer_from_config(block_config))
        for block_config in config['stage3']:
            stage3.append(set_layer_from_config(block_config))
        for block_config in config['stage4']:
            stage4.append(set_layer_from_config(block_config))

        net = TextNet(first_conv, stage1, stage2, stage3, stage4)

        return net


class FASTNeck(nn.Module):
    def __init__(self, reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4):
        super(FASTNeck, self).__init__()
        self.reduce_layer1 = reduce_layer1
        self.reduce_layer2 = reduce_layer2
        self.reduce_layer3 = reduce_layer3
        self.reduce_layer4 = reduce_layer4

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
        return F.upsample(x, size=(H, W), mode='bilinear')

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

    @staticmethod
    def build_from_config(config):
        reduce_layer1 = set_layer_from_config(config['reduce_layer1'])
        reduce_layer2 = set_layer_from_config(config['reduce_layer2'])
        reduce_layer3 = set_layer_from_config(config['reduce_layer3'])
        reduce_layer4 = set_layer_from_config(config['reduce_layer4'])
        return FASTNeck(reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4)


class FASTHead(nn.Module):
    def __init__(self, conv, blocks, final, pooling_size,
                 loss_text, loss_kernel, loss_emb, dropout_ratio=0):
        super(FASTHead, self).__init__()
        self.conv = conv
        if blocks is not None:
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = None
        self.final = final

        # self.text_loss = build_loss(loss_text)
        # self.kernel_loss = build_loss(loss_kernel)
        # self.emb_loss = build_loss(loss_emb)

        self.pooling_size = pooling_size

        self.pooling_1s = nn.MaxPool2d(kernel_size=self.pooling_size, stride=1,
                                       padding=(self.pooling_size - 1) // 2)
        self.pooling_2s = nn.MaxPool2d(kernel_size=self.pooling_size // 2 + 1, stride=1,
                                       padding=(self.pooling_size // 2) // 2)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
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
        if self.blocks is not None:
            for block in self.blocks:
                x = block(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.final(x)
        return x

    def get_results(self, out, img_meta, cfg, scale=2):

        if not self.training:
            torch.cuda.synchronize()
            start = time.time()

        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]  # 640*640
        batch_size = out.size(0)
        outputs = dict()

        texts = F.interpolate(out[:, 0:1, :, :], size=(img_size[0] // scale, img_size[1] // scale),
                              mode='nearest')  # B*1*320*320
        texts = self._max_pooling(texts, scale=scale)  # B*1*320*320
        score_maps = torch.sigmoid_(texts)  # B*1*320*320
        score_maps = F.interpolate(score_maps, size=(img_size[0], img_size[1]), mode='nearest')  # B*1*640*640
        score_maps = score_maps.squeeze(1)  # B*640*640

        kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B*160*160
        if kernels.is_cuda:
            labels_ = ccl_cuda.ccl_batch(kernels)  # B*160*160
        else:
            labels_ = []
            for kernel in kernels.numpy():
                ret, label_ = cv2.connectedComponents(kernel)
                labels_.append(label_)
            labels_ = np.array(labels_)
            labels_ = torch.from_numpy(labels_)
        labels = labels_.unsqueeze(1).to(torch.float32)  # B*1*160*160
        labels = F.interpolate(labels, size=(img_size[0] // scale, img_size[1] // scale), mode='nearest')  # B*1*320*320
        labels = self._max_pooling(labels, scale=scale)
        labels = F.interpolate(labels, size=(img_size[0], img_size[1]), mode='nearest')  # B*1*640*640
        labels = labels.squeeze(1).to(torch.int32)  # B*640*640

        keys = [torch.unique(labels_[i], sorted=True) for i in range(batch_size)]

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(
                post_time=time.time() - start
            ))

        outputs.update(dict(kernels=kernels.data.cpu()))

        scales = (float(org_img_size[1]) / float(img_size[1]),
                  float(org_img_size[0]) / float(img_size[0]))

        results = []
        for i in range(batch_size):
            bboxes, scores = generate_bbox(keys[i], labels[i], score_maps[i], scales, cfg)
            results.append(dict(
                bboxes=bboxes,
                scores=scores
            ))
        outputs.update(dict(results=results))

        return outputs

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x

    # def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances):
    #     # output
    #     kernels = out[:, 0, :, :]  # 4*640*640
    #     texts = self._max_pooling(kernels, scale=1)  # 4*640*640
    #     embs = out[:, 1:, :, :]  # 4*4*640*640
    #
    #     # text loss
    #     selected_masks = ohem_batch(texts, gt_texts, training_masks)
    #     loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
    #     iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
    #     losses = dict(
    #         loss_text=loss_text,
    #         iou_text=iou_text
    #     )
    #
    #     # kernel loss
    #     selected_masks = gt_texts * training_masks
    #     loss_kernel = self.kernel_loss(kernels, gt_kernels, selected_masks, reduce=False)
    #     loss_kernel = torch.mean(loss_kernel, dim=0)
    #     iou_kernel = iou((kernels > 0).long(), gt_kernels, selected_masks, reduce=False)
    #     losses.update(dict(
    #         loss_kernels=loss_kernel,
    #         iou_kernel=iou_kernel
    #     ))
    #
    #     # auxiliary loss
    #     loss_emb = self.emb_loss(embs, gt_instances, gt_kernels, training_masks, reduce=False)
    #     losses.update(dict(
    #         loss_emb=loss_emb
    #     ))
    #
    #     return losses

    @staticmethod
    def build_from_config(config, **kwargs):
        conv = set_layer_from_config(config['conv'])
        final = set_layer_from_config(config['final'])
        try:
            blocks = []
            for block_config in config['blocks']:
                blocks.append(set_layer_from_config(block_config))
            return FASTHead(conv, blocks, final, **kwargs)
        except:
            return FASTHead(conv, None, final, **kwargs)


class FAST(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(FAST, self).__init__()
        self.backbone = TextNet.build_from_config(backbone)
        self.neck = FASTNeck.build_from_config(neck)
        self.det_head = FASTHead.build_from_config(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs, gt_texts=None, gt_kernels=None, training_masks=None,
                gt_instances=None, img_metas=None, cfg=None):
        outputs = dict()

        if not self.training:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # reduce channel
        f = self.neck(f)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))

        if self.training:
            det_out = self._upsample(det_out, imgs.size(), scale=1)
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks, gt_instances)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), scale=4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg, scale=2)
            outputs.update(det_res)

        return outputs
