#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def add_conv_1d(x, k, reduction_fn=np.max, pad_element=-np.inf):
    """Performs additive batch convolution with a 1D kernel.
    Args:
        x: data, can be batched.
        k: kernel, must be 1D
        reduction_fn: a function for reduction in the final step. must accept `axis` argument. defaults to max.
    Returns:
        x additively convoluted with k over the last axis
    """
    k = k[::-1]
    xpk = x[..., None, :] + k[:, None]
    pad_size = k.size // 2
    pad_axes = np.zeros((len(xpk.shape), 2), dtype=np.int)
    pad_axes[-1, :] = pad_size
    xpk_pad = np.pad(xpk, pad_axes, mode='constant', constant_values=pad_element)
    xpk_shifted = np.lib.stride_tricks.as_strided(
        xpk_pad, 
        shape=xpk.shape, 
        strides=(
            xpk_pad.strides[:-2] + 
            (xpk_pad.strides[-2] + xpk_pad.strides[-1], xpk_pad.strides[-1])
        )
    )
    xpk_red = reduction_fn(xpk_shifted, axis=-2)
    return xpk_red
    
def mul_conv_1d(x, k, reduction_fn=np.max, pad_element=-np.inf, zeros=None):
    """Performs additive batch convolution with a 1D kernel.
    Args:
        x: data, can be batched.
        k: kernel, must be 1D
        reduction_fn: a function for reduction in the final step. must accept `axis` argument. defaults to max.
    Returns:
        x additively convoluted with k over the last axis
    """
    k = k[::-1]
    k = np.concatenate([np.array([0]), k, np.array([0])])
    xpk = x[..., None, :] * k[:, None] + zeros[..., None, :] * (1-k[:, None])
    pad_size = k.size // 2
    pad_axes = np.zeros((len(xpk.shape), 2), dtype=np.int)
    pad_axes[-1, :] = pad_size
    xpk_pad = np.pad(xpk, pad_axes, mode='constant', constant_values=pad_element)
    xpk_shifted = np.lib.stride_tricks.as_strided(
        xpk_pad, 
        shape=xpk.shape, 
        strides=(
            xpk_pad.strides[:-2] + 
            (xpk_pad.strides[-2] + xpk_pad.strides[-1], xpk_pad.strides[-1])
        )
    )
    xpk_red = reduction_fn(xpk_shifted, axis=-2)
    return xpk_red
    
