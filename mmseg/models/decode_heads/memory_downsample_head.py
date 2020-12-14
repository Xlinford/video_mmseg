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

    def __init__(self, matmul_norm=False, down_sample_stride=None, down_sample_method=None, align_corners=False):
        super(MemoryModule, self).__init__()
        self.matmul_norm = matmul_norm
        self.down_sample_stride = down_sample_stride
        self.align_corners = align_corners
        assert down_sample_method in ['max_pooling', 'average_pooling']
        if down_sample_method == 'max_pooling':
            self.down_sample_method = nn.MaxPool2d(kernel_size=down_sample_stride, stride=down_sample_stride,
                                                   padding=(down_sample_stride - 1) // 2)
        elif down_sample_stride == 'average_pooling':
            self.down_sample_method = nn.AvgPool2d(kernel_size=down_sample_stride, stride=down_sample_stride,
                                                   padding=(down_sample_stride - 1) // 2)

    def forward(self, memory_keys, memory_values, query_key, query_value):
        """
        Memory Module forward.
        Args:
            memory_keys (Tensor): memory keys tensor, shape: BxTxCxHxW
            memory_values (Tensor): memory values tensor, shape: BxTxCxHxW
            query_key (Tensor): query keys tensor, shape: BxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW

        Returns:
            Concat query and memory tensor.
        """
        batch_size, _, _, ori_height, ori_width = memory_keys.shape
        if self.down_sample_method is not None:
            memory_keys = [self.down_sample_method(inputs).unsqueeze(0) for inputs in memory_keys]  # B, TxCxHxW
            memory_keys = torch.cat(memory_keys, dim=0)  # BxTxCxHxW
            memory_values = [self.down_sample_method(inputs).unsqueeze(0) for inputs in memory_values]  # B, TxCxHxW
            memory_values = torch.cat(memory_values, dim=0)  # BxTxCxHxW
            # query_key = self.down_sample_method(query_key)

        batch_size, sequence_num, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels
        assert ori_height // height == self.down_sample_stride and ori_width // width == self.down_sample_stride, \
            f'ori: {ori_height, ori_width}, pooled: {height, width}'

        memory_keys = memory_keys.permute(0, 2, 1, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W
        memory_keys = memory_keys.permute(0, 2, 1).contiguous()  # BxT*H*WxC

        query_key = query_key.view(batch_size, key_channels, ori_height * ori_height)  # BxCxH*W
        key_attention = torch.bmm(memory_keys, query_key)  # BxT*H*WxH*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=1)  # BxT*H*WxH*W

        memory_values = memory_values.permute(0, 2, 1, 3, 4).contiguous()  # BxCxTxHxW
        memory_values = memory_values.view(batch_size, value_channels, sequence_num * height * width)
        memory = torch.bmm(memory_values, key_attention)  # BxCxH*W
        memory = memory.view(batch_size, value_channels, ori_height, ori_width)  # BxCxHxW

        # memory = resize(
        #     memory,
        #     size=(ori_height, ori_width),
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory


@HEADS.register_module()
class MemoryDownSampleHead(BaseDecodeHead):
    """Memory decoder for video semantic segmentation."""

    def __init__(self, sequence_num, memory_keys_channels, query_key_channels, down_sample_stride, down_sample_method,
                 **kwargs):
        super(MemoryDownSampleHead, self).__init__(**kwargs)
        self.sequence_num = sequence_num
        self.memory_key_conv = SequenceConv(self.in_channels, memory_keys_channels, 3, sequence_num,
                                            self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.memory_value_conv = SequenceConv(self.in_channels, memory_keys_channels * 4, 3, sequence_num,
                                              self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.query_key_conv = ConvModule(
            self.in_channels,
            query_key_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.query_value_conv = ConvModule(
            self.in_channels,
            query_key_channels * 4,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.memory_module = MemoryModule(matmul_norm=False, down_sample_stride=down_sample_stride,
                                          down_sample_method=down_sample_method, align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels,
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
        sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # B, TxCxHxW
        sequence_imgs = torch.cat(sequence_imgs, dim=0)  # BxTxCxHxW
        batch_size, sequence_num, channels, height, width = sequence_imgs.shape

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
