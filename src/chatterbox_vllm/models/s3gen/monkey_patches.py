from typing import Callable, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import lora
from diffusers.models.activations import GELU
from diffusers.models.attention_processor import Attention


def _Attention_forward_no_kwargs(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    # Add explicit known kwargs here if needed
    ip_adapter_masks: Optional[torch.Tensor] = None,
    ip_hidden_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # Only keep args we know
    kwargs = {}
    if ip_adapter_masks is not None:
        kwargs["ip_adapter_masks"] = ip_adapter_masks
    if ip_hidden_states is not None:
        kwargs["ip_hidden_states"] = ip_hidden_states
    residual = hidden_states
    temb: Optional[torch.Tensor] = None

    if self.spatial_norm is not None:
        hidden_states = self.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    else:
        batch_size = -1
        channel = -1
        height = -1
        width = -1

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.norm_cross is not None:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads

    query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    if self.norm_q is not None:
        query = self.norm_q(query)
    if self.norm_k is not None:
        key = self.norm_k(key)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if self.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / self.rescale_output_factor

    return hidden_states


def _Gelu_no_versioncheck(self: GELU, gate: torch.Tensor) -> torch.Tensor:
    # Just do the real GELU
    return F.gelu(gate, approximate=self.approximate)


def _Gelu_forward_no_versioncheck(self: GELU, hidden_states):
    hidden_states = self.proj(hidden_states)
    hidden_states = _Gelu_no_versioncheck(self, hidden_states)
    return hidden_states


def linear_forward_no_super(self: lora.LoRACompatibleLinear, hidden_states: torch.Tensor, scale: float = 1.0):
    if self.lora_layer is None:
        return F.linear(hidden_states, self.weight, self.bias)  # direct call
    else:
        out = F.linear(hidden_states, self.weight, self.bias)
        out = out + (self.lora_layer(hidden_states) * scale)
        return out


def apply_monkey_patches():
    GELU.forward = _Gelu_forward_no_versioncheck
    # Monkey-patch Attention.forward
    Attention.forward = _Attention_forward_no_kwargs
    lora.LoRACompatibleLinear.forward = linear_forward_no_super