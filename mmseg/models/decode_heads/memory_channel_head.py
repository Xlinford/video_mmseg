import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..utils import SequenceConv


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
            memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW
            memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW
            memory_keys_low (Tensor): low level memory keys tensor, shape: TxBxCxHxW
            memory_values_low (Tensor): low level memory values tensor, shape: TxBxCxHxW
            query_key_low (Tensor): low level query keys tensor, shape: BxCxHxW
        Returns:
            Concat query and memory tensor.
        """
        sequence_num, batch_size, key_channels, low_height, low_width = memory_keys_low.shape
        _, _, _, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key_low.shape[1] == key_channels and query_value.shape[1] == value_channels
        memory_keys_low = memory_keys_low.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys_low = memory_keys_low.view(batch_size, key_channels,
                                               sequence_num * low_height * low_width)  # BxCxT*H*W
        memory_keys_low = memory_keys_low.permute(0, 2, 1).contiguous()  # BxT*H*WxC

        query_key_low = query_key_low.view(batch_size, key_channels, low_height * low_width)  # BxCxH*W
        key_attention = torch.bmm(memory_keys_low, query_key_low)  # BxT*H*WxH*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=1)  # BxT*H*WxH*W

        memory_values_low = memory_values_low.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_values_low = memory_values_low.view(batch_size, value_channels, sequence_num * low_height * low_width)
        memory = torch.bmm(memory_values_low, key_attention)  # BxCxH*W
        memory = memory.view(batch_size, value_channels, low_height, low_width)  # BxCxHxW

        # global context module
        memory_keys = memory_keys.permute(1, 0, 2, 3, 4).contiguous()  # BxTxCxHxW
        g_memory_keys = memory_keys.view(batch_size, sequence_num, key_channels, height * width)  # BxTxCkxH*W
        memory_values = memory_values.permute(1, 0, 2, 3, 4).contiguous()  # BxTxCxHxW
        g_memory_values = memory_values.view(batch_size, sequence_num, value_channels, height * width)  # BxTxCvxH*W
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
class MemoryHead2(BaseDecodeHead):
    """Memory decoder for video semantic segmentation."""

    def __init__(self, sequence_num, key_channels, value_channels, **kwargs):
        super(MemoryHead2, self).__init__(**kwargs)
        self.sequence_num = sequence_num
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
        sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # T, BxCxHxW
        sequence_imgs = torch.cat(sequence_imgs, dim=0)  # BxTxCxHxW
        sequence_imgs_low = [self._transform_inputs(inputs, True).unsqueeze(0) for inputs in
                             ori_sequence_imgs]  # T, 1xBxCxHxW
        sequence_imgs_low = torch.cat(sequence_imgs_low, dim=0)  # TxBxCxHxW
        _, _, _, low_height, low_width = sequence_imgs_low.shape
        sequence_num, batch_size, channels, height, width = sequence_imgs.shape

        assert sequence_num == self.sequence_num
        memory_keys = self.memory_key_conv(sequence_imgs)
        memory_values = self.memory_value_conv(sequence_imgs)
        memory_keys_low = self.memory_key_conv_low(sequence_imgs_low)
        memory_values_low = self.memory_value_conv_low(sequence_imgs_low)
        query_value = self.query_value_conv(x)  # BxCxHxW
        query_key_low = self.query_key_conv_low(x_low)  # BxCxHxW

        # memory read
        output = self.memory_module(memory_keys, memory_values, query_value,
                                    memory_keys_low, memory_values_low, query_key_low)
        output = self.bottleneck(output)
        output = self.cls_seg(output)

        return output
