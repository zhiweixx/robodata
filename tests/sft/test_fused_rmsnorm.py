"""Compare the compiled norm's forward and VJP to the pinned native modules on CPU."""
import pytest


@pytest.mark.parametrize('width,weight_dtype', [(5120, 'bfloat16'), (256, 'bfloat16'),
                                               (128, 'bfloat16'), (128, 'float32')])
def test_fused_qwen_rmsnorm_preserves_casts_and_gradients(monkeypatch, width, weight_dtype):
    torch = pytest.importorskip('torch')
    qwen = pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')
    from robodata.sft import fused_rmsnorm as fusion

    torch.manual_seed(734)
    gated = width == 128
    mod = (qwen.Qwen3_5RMSNormGated if gated else qwen.Qwen3_5RMSNorm)(width)
    mod.to(getattr(torch, weight_dtype)).requires_grad_(False)
    mod.weight.uniform_(-1.25, 1.5)
    # Padded rows emulate the interleaved q/gate projection. Use multiple
    # tiles including a short tail, and a noncontiguous upstream gradient.
    h = torch.randn(1, 11, 3, 2 * width, dtype=torch.bfloat16)[..., :width].requires_grad_()
    dy = torch.randn(1, 3, 11, width, dtype=h.dtype).transpose(1, 2)
    if gated:
        h, dy = h.view(-1, width), dy.reshape(-1, width)
    elif width == 5120:
        h, dy = h[:, :, 0], dy[:, :, 0]
    args = [h]
    if gated:
        args.append(torch.randn_like(h).requires_grad_())
    monkeypatch.setattr(fusion, 'TILE_ELEMENTS', width * 3 * 4)
    reference = mod(*args)
    actual = fusion._fused_op()(h, mod.weight, 1e-6, args[1] if gated else None)
    assert actual.dtype == reference.dtype == torch.bfloat16
    if width == 5120:
        # Different FP32 reduction orders can straddle a BF16 rounding boundary.
        # Require adjacent BF16 values on those coordinates and bound their count.
        changed = actual != reference
        adjacent = torch.nextafter(reference[changed], actual[changed])
        torch.testing.assert_close(actual[changed], adjacent, rtol=0, atol=0)
        assert changed.sum().item() <= reference.numel() // 1000
    else:
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual, args, dy)
    reference_grads = torch.autograd.grad(reference, args, dy)
    for observed, expected in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(observed, expected, rtol=.008, atol=2e-4)
    assert mod.weight.dtype == getattr(torch, weight_dtype) and mod.weight.grad is None


def test_longest_interleaved_query_slices_before_flattening():
    torch = pytest.importorskip('torch')
    from robodata.sft import fused_rmsnorm as fusion

    projection = torch.empty((1, 242322, 24, 512), device='meta', dtype=torch.bfloat16)
    query = projection[..., :256]
    last_element = sum((n - 1) * st for n, st in zip(query.shape, query.stride(), strict=True))
    assert last_element > 2**31
    tile_sizes = [tile.numel() for (tile,) in fusion._pieces(query)]
    assert max(tile_sizes) <= fusion.TILE_ELEMENTS and sum(tile_sizes) == query.numel()
