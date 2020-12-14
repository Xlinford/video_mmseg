import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from ..builder import HEADS


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

    def forward(self, query_value, memory_keys_low, memory_values_low, query_key_low):
        """
        Memory Module forward.
        Args:
            query_value (Tensor): query values tensor, shape: BxCxHxW
            memory_keys_low (Tensor): low level memory keys tensor, shape: BxTxCxHxW
            memory_values_low (Tensor): low level memory values tensor, shape: BxTxCxHxW
            query_key_low (Tensor): low level query keys tensor, shape: BxCxHxW
        Returns:
            Concat query and memory tensor.
        """
        batch_size, sequence_num, key_channels, low_height, low_width = memory_keys_low.shape
        _, _, value_channels, _, _ = memory_values_low.shape
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

        query_value = resize(
            query_value,
            size=(low_height, low_width),
            mode='bilinear',
            align_corners=self.align_corners)

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory


@HEADS.register_module()
class MemoryL8Head(BaseDecodeHead):
    """Memory decoder for video semantic segmentation."""

    def __init__(self, sequence_num, key_channels, value_channels, **kwargs):
        super(MemoryL8Head, self).__init__(**kwargs)
        self.sequence_num = sequence_num
        self.memory_key_conv_low = SequenceConv(self.in_channels // 4, key_channels, 3, sequence_num,
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
        memory_keys_low = self.memory_key_conv_low(sequence_imgs_low)
        memory_values_low = self.memory_value_conv_low(sequence_imgs_low)
        query_value = self.query_value_conv(x)  # BxCxHxW
        query_key_low = self.query_key_conv_low(x_low)  # BxCxHxW

        # memory read
        output = self.memory_module(query_value, memory_keys_low, memory_values_low, query_key_low)
        output = self.bottleneck(output)
        output = self.cls_seg(output)

        return output
