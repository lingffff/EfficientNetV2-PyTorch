# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: EfficientNet V2 model.
# Create: 2021-12-2
import torch
import torch.nn as nn
from utils import get_efficientnetv2_params


# CBAM Module
# Not use, just for practice.
class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self._r = reduction
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)
        self._fc1 = nn.Linear(channels, channels // self._r, bias=False)
        self._relu = nn.ReLU(inplace=True)
        self._fc2 = nn.Linear(channels // self._r, channels, bias=False)
        self._sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, _, _ = inputs.size()
        x1 = self._avg_pool(inputs).squeeze()
        x2 = self._max_pool(inputs).squeeze()

        x1 = self._fc2(self._relu(self._fc1(x1)))
        x2 = self._fc2(self._relu(self._fc1(x2)))
      
        y = self._sigmoid(x1 + x2)
        return inputs * y.view(b, c, 1, 1).expand_as(inputs)


# CBAM Module
# Not use, just for practice.
class SpartialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self._max_pool = nn.AdaptiveMaxPool2d((None, 1))
        padding = (kernel_size - 1) // 2
        self._conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self._sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, _, h, w = inputs.size()
        trans_inputs = inputs.transpose(1, 3)
        x1 = self._avg_pool(trans_inputs).transpose(1, 3)
        x2 = self._max_pool(trans_inputs).transpose(1, 3)
        x = torch.cat((x1, x2), dim=1)
        x = self._sigmoid(self._conv(x))
        return inputs * x.view(b, 1, h, w).expand_as(inputs)


class SEModule(nn.Module):
    def __init__(self, channels, ratio=1/16):
        super().__init__()
        self._r = ratio
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = int(channels * self._r)
        self._fc1 = nn.Linear(channels, hidden_channels, bias=False)
        self._relu = nn.ReLU(inplace=True)
        self._fc2 = nn.Linear(hidden_channels, channels, bias=False)
        self._sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, _, _ = inputs.size()
        x = self._avg_pool(inputs).squeeze()
        x = self._relu(self._fc1(x))
        x = self._sigmoid(self._fc2(x))
        return inputs * x.view(b, c, 1, 1).expand_as(inputs)


class Conv2dAutoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        assert kernel_size % 2 == 1, "Only support odd kernel size."
        padding = (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=bias)


class MBConvBlock(nn.Module):
    def __init__(self, block_arg):
        super().__init__()
        self._block_arg = block_arg
        # expand
        inc = self._block_arg.input_filters
        outc = inc * self._block_arg.expand_ratio
        if self._block_arg.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(inc, outc, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(outc)
        # dw
        self._dw_conv = Conv2dAutoPadding(outc, outc, self._block_arg.kernel_size, 
                                    self._block_arg.stride, groups=outc, bias=False)
        self._bn1 = nn.BatchNorm2d(outc)
        # squeeze and extract
        if self._block_arg.se_ratio:
            self._se = SEModule(outc, self._block_arg.se_ratio)
        # pw
        inc = outc
        outc = self._block_arg.output_filters
        self._pw_conv = nn.Conv2d(inc, outc, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(outc)
        # activation
        self._swish = nn.SiLU()

    def forward(self, inputs):
        x = inputs
        if self._block_arg.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._dw_conv(x)))
        if self._block_arg.se_ratio:
            x = self._se(x)
        x = self._bn2(self._pw_conv(x))  # pw conv: linear activation
        if self._block_arg.input_filters == self._block_arg.output_filters and self._block_arg.stride == 1:
            x = x + inputs
        return x


class FusedMBConvBlock(nn.Module):
    def __init__(self, block_arg):
        super().__init__()
        self._block_arg = block_arg
        # fused conv
        inc = self._block_arg.input_filters
        outc = inc * self._block_arg.expand_ratio
        self._fused_conv = Conv2dAutoPadding(inc, outc, self._block_arg.kernel_size, self._block_arg.stride, bias=False)
        self._bn = nn.BatchNorm2d(outc)
        # squeeze and extract
        if self._block_arg.se_ratio:
            self._se = SEModule(outc, self._block_arg.se_ratio)
        # pw
        inc = outc
        outc = self._block_arg.output_filters
        self._pw_conv = nn.Conv2d(inc, outc, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(outc)
        # activation
        self._swish = nn.SiLU()

    def forward(self, inputs):
        x = inputs
        x = self._swish(self._bn(self._fused_conv(inputs)))
        if self._block_arg.se_ratio:
            x = self._se(x)
        x = self._bn2(self._pw_conv(x))  # pw conv: linear activation
        if self._block_arg.input_filters == self._block_arg.output_filters and self._block_arg.stride == 1:
            x = x + inputs
        return x


class EfficientNetV2(nn.Module):
    def __init__(self, blocks_args, global_params):
        super().__init__()
        self._blocks_args = blocks_args
        self._global_params = global_params
        # stem
        inc = 3
        outc = blocks_args[0].input_filters
        self._stem_conv = Conv2dAutoPadding(inc, outc, 3, 2)
        self._bn0 = nn.BatchNorm2d(outc)
        # blocks
        self._blocks = nn.ModuleList([]) # BUG: [] -> nn.ModuleList([])
        for block_arg in self._blocks_args:
            block = FusedMBConvBlock(block_arg) if block_arg.fused == True else MBConvBlock(block_arg)
            self._blocks.append(block)
            if block_arg.num_repeat > 1:
                block_arg = block_arg._replace(input_filters=block_arg.output_filters, stride=1)
            for _ in range(block_arg.num_repeat - 1):
                block = FusedMBConvBlock(block_arg) if block_arg.fused == True else MBConvBlock(block_arg)
                self._blocks.append(block)
        # head
        inc = block_arg.output_filters
        outc = int(self._global_params.width_coefficient * 1280)
        self._head_conv = nn.Conv2d(inc, outc, 1, 1)
        self._bn1 = nn.BatchNorm2d(outc)
        # top
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate) # missing dropout
        self._fc = nn.Linear(outc, self._global_params.num_classes)
        # activation
        self._swish = nn.SiLU()  # hasattr?


    def forward(self, inputs):
        x = self._swish(self._bn0(self._stem_conv(inputs)))

        for i, block in enumerate(self._blocks): # BUG: missing enumerate
            x = block(x)
        
        x = self._swish(self._bn1(self._head_conv(x)))

        x = self._avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x



if __name__ == '__main__':
    blocks_args, global_params = get_efficientnetv2_params('efficientnetv2-b0', 1000)
    model = EfficientNetV2(blocks_args, global_params)
    image_size = global_params.image_size
    x = torch.randn(1, 3, image_size, image_size)
    print("Input shape:", x.shape)
    y = model(x)
    print("Output shape:", y.shape)   
