"""Fuse only the eager BF16 SiLU and multiply inside Qwen's language MLP."""
from types import MethodType

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ModuleNotFoundError as exc:
    if exc.name != 'triton':
        raise
    triton = None

BLOCK = 1024
MODE = 'triton_eager_bf16_swiglu_v1'


if triton is not None:
    @triton.jit
    def _forward_kernel(gate, up, out, count, BLOCK_SIZE: tl.constexpr):
        # The longest real projection exceeds 2^31 elements. Widen the program
        # index *before* multiplying, even when the launch grid itself is small.
        offset = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offset < count
        x = tl.load(gate + offset, mask=valid, other=0.).to(tl.float32)
        y = tl.load(up + offset, mask=valid, other=0.).to(tl.float32)
        # Native CUDA SiLU divides in float32 and stores BF16 before multiply.
        activated = tl.div_rn(x, 1.0 + libdevice.exp(-x)).to(tl.bfloat16).to(tl.float32)
        tl.store(out + offset, activated * y, mask=valid)

    @triton.jit
    def _backward_kernel(gate, up, grad, dgate, dup, count, rows, cols,
                         batch_stride, row_stride, col_stride,
                         GRAD_CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offset = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offset < count
        x = tl.load(gate + offset, mask=valid, other=0.).to(tl.float32)
        y = tl.load(up + offset, mask=valid, other=0.).to(tl.float32)
        if GRAD_CONTIGUOUS:
            grad_offset = offset
        else:
            # Read expanded or strided upstream gradients without allocating
            # another full tensor just to make it contiguous.
            # offset/row are already signed int64; every stride product widens
            # before multiplication, including when a stride specializes to 0 or 1.
            row = offset // cols
            grad_offset = ((row // rows) * batch_stride
                           + (row % rows) * row_stride
                           + (offset % cols) * col_stride)
        g = tl.load(grad + grad_offset, mask=valid, other=0.).to(tl.float32)
        denom = 1.0 + libdevice.exp(-x)
        sigmoid = tl.div_rn(1.0, denom)
        activated = tl.div_rn(x, denom).to(tl.bfloat16).to(tl.float32)
        # Eager mul backward stores BF16 g*y as the incoming SiLU gradient.
        dsilu = (g * y).to(tl.bfloat16).to(tl.float32)
        dx = (dsilu * sigmoid) * tl.fma(x, 1.0 - sigmoid, 1.0)
        tl.store(dgate + offset, dx, mask=valid)
        tl.store(dup + offset, g * activated, mask=valid)


def _grid(count):
    programs = (count + BLOCK - 1) // BLOCK
    if programs > 2**31 - 1:
        raise ValueError(f'One-dimensional CUDA grid is too large: {programs}')
    return (programs,)


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if (not gate.is_cuda or gate.device != up.device or gate.dtype != torch.bfloat16
                or up.dtype != gate.dtype or up.shape != gate.shape or gate.ndim not in (2, 3)
                or not gate.is_contiguous() or not up.is_contiguous()):
            raise ValueError('Expected equal contiguous 2D/3D CUDA BF16 projection outputs')
        out = torch.empty_like(gate)
        if gate.numel():
            with torch.cuda.device(gate.device):
                _forward_kernel[_grid(gate.numel())](gate, up, out, gate.numel(), BLOCK_SIZE=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        if gate.numel():
            batch_stride = grad.stride(0) if grad.ndim == 3 else 0
            with torch.cuda.device(gate.device):
                _backward_kernel[_grid(gate.numel())](
                    gate, up, grad, dgate, dup, gate.numel(), gate.size(-2), gate.size(-1),
                    batch_stride, grad.stride(-2), grad.stride(-1),
                    GRAD_CONTIGUOUS=grad.is_contiguous(), BLOCK_SIZE=BLOCK)
        return dgate, dup


def _fused_forward(mlp, hidden):
    gate = mlp.gate_proj(hidden)
    up = mlp.up_proj(hidden)
    return mlp.down_proj(_FusedSwiGLU.apply(gate, up))


def install_fused_swiglu(model):
    """Enable the pointwise kernel on the pinned model's 64 unmodified language MLPs."""
    from transformers.activations import SiLUActivation
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    if triton is None:
        raise RuntimeError('The fused language SwiGLU requires the CUDA Triton dependency')
    mlps = [layer.mlp for layer in model.get_base_model().model.language_model.layers]
    if len(mlps) != 64 or any(type(mlp) is not Qwen3_5MLP or type(mlp.act_fn) is not SiLUActivation
                             or mlp.config.hidden_act != 'silu' or 'forward' in vars(mlp)
                             for mlp in mlps):
        raise ValueError('Expected 64 unmodified SiLU language MLPs of the pinned Qwen model')
    for mlp in mlps:
        mlp.forward = MethodType(_fused_forward, mlp)
    return {'swiglu_mode': MODE, 'fused_language_mlps': len(mlps),
            'index_bits': 64, 'elementwise_block_size': BLOCK}
