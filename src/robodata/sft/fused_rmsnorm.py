"""Opt-in, frozen-weight RMSNorm fusion for the pinned Qwen language decoder."""
from collections import Counter
from functools import cache
from math import prod
from types import MethodType


# Limit each compiler invocation to this many elements. A large gate or a
# strided query can span billions of elements across the complete trajectory.
TILE_ELEMENTS = 1 << 24


def _rows(x, weight, eps, gate):
    import torch
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if gate is None:
        return (normalized * (1.0 + weight.float())).to(x.dtype)
    # The pinned GatedDeltaNet rounds the normalization before the weight,
    # then applies SiLU in FP32. Preserve the multiply's native result dtype.
    weighted = weight * normalized.to(x.dtype)
    return (weighted * torch.nn.functional.silu(gate.float())).to(x.dtype)


def _forward_tile(x, weight, eps, gate, out):
    out.copy_(_rows(x, weight, eps, gate))


def _backward_tile(x, weight, eps, gate, dy, dx, dg):
    import torch
    # Differentiate the pinned casts as well as the reductions. The frozen
    # weight is captured, so no weight-gradient reduction is constructed.
    if gate is None:
        _, vjp = torch.func.vjp(lambda v: _rows(v, weight, eps, None), x)
        (gx,) = vjp(dy)
    else:
        _, vjp = torch.func.vjp(lambda v, g: _rows(v, weight, eps, g), x, gate)
        gx, gg = vjp(dy)
        dg.copy_(gg)
    dx.copy_(gx)


def _pieces(x, *tensors):
    # Inputs are [1,T,D], [1,T,H,D], or [T*H,D]. Slice before reshaping
    # so noncontiguous query views / upstream gradients are never copied in full.
    matrix = x.ndim == 2
    if not matrix and (x.ndim not in (3, 4) or x.shape[0] != 1):
        raise ValueError(f'Expected a single language sequence or flattened heads, got {x.shape}')
    first = x if matrix else x[0]
    tile_rows = TILE_ELEMENTS // prod(first.shape[1:])
    if not tile_rows:
        raise ValueError('A single token exceeds the language norm tile limit')
    seqs = [None if t is None else (t if matrix else t[0]) for t in (x, *tensors)]
    width = x.shape[-1]
    for start in range(0, len(first), tile_rows):
        # Only BF16 tile-sized copies can be needed here. Contiguous caller
        # outputs become views. Storage offsets are handled by PyTorch before
        # dispatch, so all compiler-visible indexing fits inside the tile.
        yield tuple(None if t is None else t[start:start + tile_rows].contiguous().view(-1, width)
                    for t in seqs)


@cache
def _fused_op():
    import torch
    opts = {'emulate_precision_casts': True}
    forward = torch.compile(_forward_tile, dynamic=True, fullgraph=True, options=opts)
    backward = torch.compile(_backward_tile, dynamic=True, fullgraph=True, options=opts)

    class FrozenRMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps, gate):
            if x.dtype != torch.bfloat16 or weight.requires_grad:
                raise ValueError('Expected BF16 language activations and a frozen norm weight')
            if gate is not None and (gate.shape != x.shape or gate.dtype != x.dtype):
                raise ValueError('Unexpected GatedDeltaNet gate shape or dtype')
            out = x.new_empty(x.shape)
            for x_tile, gate_tile, out_tile in _pieces(x, gate, out):
                forward(x_tile, weight, eps, gate_tile, out_tile)
            ctx.save_for_backward(x, weight, gate)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, dy):
            x, weight, gate = ctx.saved_tensors
            dx = x.new_empty(x.shape)
            dg = gate.new_empty(gate.shape) if gate is not None else None
            for x_tile, gate_tile, dy_tile, dx_tile, dg_tile in _pieces(x, gate, dy, dx, dg):
                backward(x_tile, weight, ctx.eps, gate_tile, dy_tile, dx_tile, dg_tile)
            return dx, None, None, dg

    return FrozenRMSNorm.apply


def _plain(self, x):
    return _fused_op()(x, self.weight, self.eps, None)


def _gated(self, hidden_states, gate):
    return _fused_op()(hidden_states, self.weight, self.variance_epsilon, gate)


def install_fused_rmsnorm(model):
    """Patch only the 161 zero-centered and 48 gated norms on this PEFT model."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated

    decoder = model.get_base_model().model.language_model
    targets = [('final', decoder.norm, Qwen3_5RMSNorm, 5120)]
    if len(decoder.layers) != 64:
        raise ValueError('Expected 64 language decoder layers')
    for layer in decoder.layers:
        for name in ('input_layernorm', 'post_attention_layernorm'):
            targets.append((name, getattr(layer, name), Qwen3_5RMSNorm, 5120))
        if layer.block_type == 'full_attention':
            for name in ('q_norm', 'k_norm'):
                targets.append((name, getattr(layer.self_attn, name), Qwen3_5RMSNorm, 256))
        elif layer.block_type == 'linear_attention':
            targets.append(('gated', layer.linear_attn.norm, Qwen3_5RMSNormGated, 128))
        else:
            raise ValueError(f'Unexpected decoder block type {layer.block_type!r}')
    counts = Counter(name for name, *_ in targets)
    if counts != dict(final=1, input_layernorm=64, post_attention_layernorm=64, q_norm=16, k_norm=16, gated=48):
        raise ValueError(f'Unexpected pinned decoder norm counts: {counts}')

    if len({id(mod) for _, mod, _, _ in targets}) != len(targets):
        raise ValueError('Unexpected shared language norm module')
    for name, mod, cls, width in targets:
        gated = name == 'gated'
        if (type(mod) is not cls or 'forward' in vars(mod) or mod.weight.shape != (width,)
                or mod.weight.requires_grad or mod.weight.dtype not in (torch.bfloat16, torch.float32)
                or not mod.weight.is_cuda or (gated and mod.activation != 'silu')
                or (mod.variance_epsilon if gated else mod.eps) != 1e-6):
            raise ValueError(f'Unexpected frozen language norm at {name}')
    for name, mod, _, _ in targets:
        mod.forward = MethodType(_gated if name == 'gated' else _plain, mod)
    return {'mode': 'torch_inductor_frozen_tiled_rmsnorm', 'tile_elements': TILE_ELEMENTS,
            'norm_counts': dict(counts), 'master_weight_dtypes': dict(Counter(str(m.weight.dtype)
                                                                             for _, m, _, _ in targets))}
