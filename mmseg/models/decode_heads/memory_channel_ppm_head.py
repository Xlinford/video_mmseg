import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from ..builder import HEADS


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

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

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
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
        """Forward function."""
        max_scale = max(self.pool_scales)
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=(max_scale, max_scale),
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class SequenceConv(nn.ModuleList):
    """Sequence conv module.

    Args:
        in_channels (int): input tensor channel.
        out_channels (int): output tensor channel.
        kernel_size (int): convolution kernel size.
        sequence_num (int): sequence length.
        conv_cfg (dict): convolution config dictionary.
        norm_cfg (dict): normalization config dictionary.
        act_cfg (dict): activation config dictionary.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sequence_num, conv_cfg, norm_cfg, act_cfg):
        super(SequenceConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sequence_num = sequence_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for _ in range(sequence_num):
            self.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    padding=self.kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            )

    def forward(self, sequence_imgs):
        """

        Args:
            sequence_imgs (Tensor): BxTxCxHxW

        Returns:
            sequence conv output: BxTxCxHxW
        """
        sequence_outs = []
        assert sequence_imgs.shape[1] == self.sequence_num
        sequence_imgs = sequence_imgs.permute(1, 0, 2, 3, 4).contiguous()  # TxBxCxHxW
        for i, sequence_conv in enumerate(self):
            sequence_out = sequence_conv(sequence_imgs[i, ...])
            sequence_out = sequence_out.unsqueeze(0)
            sequence_outs.append(sequence_out)

        sequence_outs = torch.cat(sequence_outs, dim=0).permute(1, 0, 2, 3, 4).contiguous()  # BxTxCxHxW
        return sequence_outs


class MemoryModule(nn.Module):
    """Memory read module.
    Args:

    """

    def __init__(self, matmul_norm=False, align_corners=False):
        super(MemoryModule, self).__init__()
        self.matmul_norm = matmul_norm
        self.align_corners = align_corners

    def forward(self, memory_keys, memory_values, query_value,
                memory_keys_low, memory_values_low, query_key_low):
        """
        Memory Module forward.
        Args:
            memory_keys (Tensor): memory keys tensor, shape: BxTxCxHxW
            memory_values (Tensor): memory values tensor, shape: BxTxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW
            memory_keys_low (Tensor): low level memory keys tensor, shape: BxTxCxHxW
            memory_values_low (Tensor): low level memory values tensor, shape: BxTxCxHxW
            query_key_low (Tensor): low level query keys tensor, shape: BxCxHxW
        Returns:
            Concat query and memory tensor.
        """
        batch_size, sequence_num, key_channels, low_height, low_width = memory_keys_low.shape
        batch_size, sequence_num, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key_low.shape[1] == key_channels and query_value.shape[1] == value_channels
        memory_keys_low = memory_keys_low.permute(0, 2, 1, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys_low = memory_keys_low.view(batch_size, key_channels,
                                               sequence_num * low_height * low_width)  # BxCxT*H*W
        memory_keys_low = memory_keys_low.permute(0, 2, 1).contiguous()  # BxT*H*WxC

        query_key_low = query_key_low.view(batch_size, key_channels, low_height * low_width)  # BxCxH*W
        key_attention = torch.bmm(memory_keys_low, query_key_low)  # BxT*H*WxH*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=1)  # BxT*H*WxH*W

        memory_values_low = memory_values_low.permute(0, 2, 1, 3, 4).contiguous()  # BxCxTxHxW
        memory_values_low = memory_values_low.view(batch_size, value_channels, sequence_num * low_height * low_width)
        memory = torch.bmm(memory_values_low, key_attention)  # BxCxH*W
        memory = memory.view(batch_size, value_channels, low_height, low_width)  # BxCxHxW

        # global context module
        g_memory_keys = memory_keys.view(batch_size, sequence_num, key_channels, height * width)  # BxTxCk*H*W
        g_memory_values = memory_values.view(batch_size, sequence_num, value_channels, height * width)  # BxTxCv*H*W
        g_memory_values = g_memory_values.permute(0, 1, 3, 2).contiguous()  # BxTxH*WxCv
        g_memory_attention = [torch.bmm(keys, values).unsqueeze(0) for keys, values in
                              zip(g_memory_keys, g_memory_values)]
        g_memory_attention = torch.cat(g_memory_attention, dim=0)  # BxTxCkxCv
        if self.matmul_norm:
            g_memory_attention = (value_channels ** -.5) * g_memory_attention
        g_memory_attention = F.softmax(g_memory_attention, dim=1)  # BxTxCkxCv

        query_value = query_value.view(batch_size, value_channels, height * width)  # BxCvxH*W
        batch_query_value = []
        for _batch_memory_attention, _batch_query_value in zip(g_memory_attention, query_value):
            sequence_query_value = [torch.bmm(memory_attention.unsqueeze(0), _batch_query_value.unsqueeze(0)) for
                                    memory_attention in _batch_memory_attention]
            batch_query_value.append(torch.cat(sequence_query_value, dim=1))  # B, 1xT*CkxH*W

        query_value = torch.cat(batch_query_value, dim=0)  # BxT*CkxH*W
        query_value = query_value.view(batch_size, sequence_num * key_channels, height, width)  # BxT*CkxH*W

        query_value = resize(
            query_value,
            size=(low_height, low_width),
            mode='bilinear',
            align_corners=self.align_corners)

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory


@HEADS.register_module()
class MemoryChannelPPMHead(BaseDecodeHead):
    """Memory decoder for video semantic segmentation."""

    def __init__(self, sequence_num, key_channels, value_channels, pool_scales=(1, 2, 3, 6), **kwargs):
        super(MemoryChannelPPMHead, self).__init__(**kwargs)
        self.sequence_num = sequence_num
        self.pool_scales = pool_scales
        self.memory_value_ppm = PPM(pool_scales, value_channels, value_channels // len(pool_scales),
                                    self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners)
        self.memory_key_ppm = PPM(pool_scales, key_channels, key_channels // len(pool_scales),
                                  self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners)
        self.memory_key_conv = SequenceConv(self.in_channels, key_channels, 3, sequence_num,
                                            self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.memory_key_conv_low = SequenceConv(self.in_channels // 4, key_channels, 3, sequence_num,
                                                self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.memory_value_conv = SequenceConv(self.in_channels, value_channels, 3, sequence_num,
                                              self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.memory_value_conv_low = SequenceConv(self.in_channels // 4, value_channels, 3, sequence_num,
                                                  self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.query_value_conv = ConvModule(
            self.in_channels,
            value_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.query_key_conv_low = ConvModule(
            self.in_channels // 4,
            key_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.memory_module = MemoryModule(matmul_norm=False, align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            key_channels * self.sequence_num + value_channels,
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
            sequence_imgs (list[Tensor]): len(sequence_imgs) is equal to batch_size,
                each element is a Tensor with shape of TxCxHxW.

        Returns:
            decoder logits.
        """
        x = self._transform_inputs(inputs)
        x_low = self._transform_inputs(inputs, True)
        ori_sequence_imgs = sequence_imgs
        sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # B, TxCxHxW
        sequence_imgs = torch.cat(sequence_imgs, dim=0)  # BxTxCxHxW
        sequence_imgs_low = [self._transform_inputs(inputs, True).unsqueeze(0) for inputs in
                             ori_sequence_imgs]  # B, TxCxHxW
        sequence_imgs_low = torch.cat(sequence_imgs_low, dim=0)  # BxTxCxHxW
        _, _, _, low_height, low_width = sequence_imgs_low.shape
        batch_size, sequence_num, channels, height, width = sequence_imgs.shape

        assert sequence_num == self.sequence_num
        memory_keys = self.memory_key_conv(sequence_imgs)
        memory_values = self.memory_value_conv(sequence_imgs)
        memory_keys_low = self.memory_key_conv_low(sequence_imgs_low)
        memory_keys_low = self.memory_key_ppm(memory_keys_low)
        memory_values_low = self.memory_value_conv_low(sequence_imgs_low)
        memory_values_low = self.memory_value_ppm(memory_values_low)
        query_value = self.query_value_conv(x)  # BxCxHxW
        query_key_low = self.query_key_conv_low(x_low)  # BxCxHxW

        # memory read
        output = self.memory_module(memory_keys, memory_values, query_value,
                                    memory_keys_low, memory_values_low, query_key_low)
        output = self.bottleneck(output)
        output = self.cls_seg(output)

        return output
