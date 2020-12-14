import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..utils import SequenceConv


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg, act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function.
        x: BxCxHxW
        """
        ppm_outs = []
        pool_maxsize = max(self.pool_scales)
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=(pool_maxsize, pool_maxsize),
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)  # BxCxHxW
        ppm_outs = torch.cat(ppm_outs, dim=1)  # BxC*LxHxW

        return ppm_outs


class SequencePPM(nn.ModuleList):
    """Sequence Pooling Pyramid Module used.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, sequence_num, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(SequencePPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.sequence_num = sequence_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for _ in range(sequence_num):
            self.append(
                PPM(pool_scales, in_channels, channels, conv_cfg, norm_cfg, act_cfg, align_corners)
            )

    def forward(self, sequence_imgs):
        """Forward function."""
        sequence_ppm_outs = []
        assert sequence_imgs.shape[0] == self.sequence_num  # TxBxCxHxW
        for i, sequence_ppm in enumerate(self):
            ppm_outs = sequence_ppm(sequence_imgs[i, ...])
            sequence_ppm_outs.append(ppm_outs.unsqueeze(0))  # T, 1xBxCxHxW
        sequence_ppm_outs = torch.cat(sequence_ppm_outs, dim=0)  # TxBxCxHxW

        return sequence_ppm_outs


class MemoryModule(nn.Module):
    """Memory read module.
    Args:

    """

    def __init__(self, matmul_norm, pool_scales, key_channels, value_channels, sequence_num,
                 conv_cfg, norm_cfg, act_cfg, align_corners):
        super(MemoryModule, self).__init__()
        self.matmul_norm = matmul_norm
        self.memory_key_ppm = SequencePPM(pool_scales, key_channels, key_channels // len(pool_scales), sequence_num,
                                          conv_cfg, norm_cfg, act_cfg, align_corners)
        self.memory_value_ppm = SequencePPM(pool_scales, value_channels, value_channels // len(pool_scales),
                                            sequence_num, conv_cfg, norm_cfg, act_cfg, align_corners)

    def forward(self, memory_keys, memory_values, query_key, query_value):
        """
        Memory Module forward.
        Args:
            memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW
            memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
            query_key (Tensor): query keys tensor, shape: BxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW

        Returns:
            Concat query and memory tensor.
        """

        sequence_num, batch_size, key_channels, ori_height, ori_width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels

        memory_keys = self.memory_key_ppm(memory_keys)  # TxBxCxHxW
        memory_values = self.memory_value_ppm(memory_values)    # TxBxCxHxW
        _, _, _, height, width = memory_values.shape

        memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W
        memory_keys = memory_keys.permute(0, 2, 1).contiguous()  # BxT*H*WxC

        query_key = query_key.view(batch_size, key_channels, ori_height * ori_width)  # BxCxH*W
        key_attention = torch.bmm(memory_keys, query_key)  # BxT*H*WxH*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=1)  # BxT*H*WxH*W

        memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_values = memory_values.view(batch_size, value_channels, sequence_num * height * width)
        memory = torch.bmm(memory_values, key_attention)  # BxCxH*W
        memory = memory.view(batch_size, value_channels, ori_height, ori_width)  # BxCxHxW

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory


@HEADS.register_module()
class MemoryPPMHead(BaseDecodeHead):
    """Memory decoder for video semantic segmentation."""

    def __init__(self, sequence_num, key_channels, value_channels, pool_scales=(1, 2, 3, 6), **kwargs):
        super(MemoryPPMHead, self).__init__(**kwargs)
        self.sequence_num = sequence_num
        self.pool_scales = pool_scales
        self.memory_key_conv = SequenceConv(self.in_channels, key_channels, 3, sequence_num,
                                            self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.memory_value_conv = SequenceConv(self.in_channels, value_channels, 3, sequence_num,
                                              self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.query_key_conv = ConvModule(
            self.in_channels,
            key_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.query_value_conv = ConvModule(
            self.in_channels,
            value_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.memory_module = MemoryModule(False, self.pool_scales, key_channels, value_channels, sequence_num,
                                          self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners)
        self.bottleneck = ConvModule(
            value_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, inputs, sequence_imgs):
        """
        Forward fuction.
        Args:
            inputs (list[Tensor]): backbone multi-level outputs.
            sequence_imgs (list[Tensor]): len(sequence_imgs) is equal to sequence,
                each element is a Tensor with shape of BxCxHxW.

        Returns:
            decoder logits.
        """
        x = self._transform_inputs(inputs)
        sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # T, BxCxHxW
        sequence_imgs = torch.cat(sequence_imgs, dim=0)  # TxBxCxHxW
        sequence_num, batch_size, channels, height, width = sequence_imgs.shape

        assert sequence_num == self.sequence_num
        memory_keys = self.memory_key_conv(sequence_imgs)
        memory_values = self.memory_value_conv(sequence_imgs)
        query_key = self.query_key_conv(x)  # BxCxHxW
        query_value = self.query_value_conv(x)  # BxCxHxW

        # memory read
        output = self.memory_module(memory_keys, memory_values, query_key, query_value)
        output = self.bottleneck(output)
        output = self.cls_seg(output)

        return output
