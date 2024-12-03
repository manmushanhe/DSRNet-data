#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

class Frequency_Attention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = n_feat 
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        


    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, channels, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        #n_batch = query.size(0)
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        q = q.transpose(1, 2)  # (batch, channels, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, channels, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, channels, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, channels, time1, d_model)
                weighted by the attention score (#batch, channels, time1, time2).

        """
        mask = None
        if mask is not None:
            mask = mask.eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)   # (batch, channels, time1, d_k)


        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  
        x = x.transpose(1, 2)
        # (batch, channels, time1, d_k)

        return self.linear_out(x)  # (batch, channels, time1, d_k)

    def forward(self, feat):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        #mask = (~make_pad_mask(feats_len)[:, None, :]).to(feat.device)
        
        
        q, k, v = self.forward_qkv(feat, feat, feat)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        output = self.forward_attention(v, scores)
        return output

