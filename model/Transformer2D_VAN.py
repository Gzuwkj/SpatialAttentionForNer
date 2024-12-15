import json

import torch
from torch import nn
from torch.functional import F
import numpy as np

import math

def angle(a, b):
    a_norm = a / torch.norm(a, dim=-1, keepdim=True)
    b_norm = b / torch.norm(b, dim=-1, keepdim=True)
    cosine = torch.sum(a_norm * b_norm, dim=-1)
    angle = torch.acos(cosine)
    degree = angle * 180 / math.pi
    return -degree + 90

class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """
        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class T2dLocalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_proj = nn.Sequential(
            # nn.ZeroPad2d(padding=(0, 1, 1, 0)),
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=config.kernel_size,
                padding=config.kernel_size//2,
                groups=config.hidden_size
            ),
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=1,
                padding=0,
                groups=1
            ),
        )

    def forward(
            self,
            span_states,
            attention_mask=None,
            output_attentions=False
    ):
        span_states = torch.masked_fill(span_states, attention_mask.eq(0).unsqueeze(1), 0.0)
        attention_prob = self.conv_proj(span_states)
        if attention_mask is not None:
            attention_prob = attention_prob * attention_mask.unsqueeze(1)
        context_layer = attention_prob * span_states, None
        if output_attentions:
            ang = angle(attention_prob.permute(0, 2, 3, 1), span_states.permute(0, 2, 3, 1))
            context_layer += ang,
        return context_layer


class T2dLayer(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.local_attention = T2dLocalAttention(config)
        self.LayerNormOut = LayerNorm((1, config.hidden_size, 1, 1), dim_index=1)
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=1,
                groups=1,
            ),
            nn.GELU()
        )

    def forward(
            self,
            span_states,
            attention_mask=None,
            output_attentions=False,
    ):
        _span_states = span_states
        span_states = self.local_attention(
            span_states,
            attention_mask,
            output_attentions
        )
        if output_attentions:
            attention = span_states[-1]
        span_states = span_states[0]
        __span_states = span_states + _span_states
        span_states = self.LayerNormOut(__span_states) + _span_states
        span_states = self.out(span_states)
        span_states = (span_states, )
        if output_attentions:
            span_states += attention,
        return span_states

class T2d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([T2dLayer(config, idx) for idx in range(config.num_hidden_layers)])

    def forward(
            self,
            span_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        all_span_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_span_states = all_span_states + (span_states,)
            layer_outputs = layer_module(
                span_states,
                attention_mask,
                output_attentions,
            )
            span_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_hidden_states:
                all_span_states = all_span_states + (span_states,)
        return tuple(
            v
            for v in [span_states, all_span_states, all_self_attentions]
            if v is not None
        )


class T2dConfig(object):
    def __init__(self, path):
        dic = json.loads(open(path, 'r').read())
        self.__dict__.update(dic)
