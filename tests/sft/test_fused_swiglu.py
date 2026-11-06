"""Check the launch bounds, the BF16 rounding contract and the untiled projection wiring."""
import copy

import pytest

torch = pytest.importorskip('torch')

from robodata.sft import fused_swiglu


def test_largest_sequence_uses_legal_1d_cuda_grid():
    count = 242_322 * 17_408
    blocks, = fused_swiglu._grid(count)
    assert blocks * fused_swiglu.BLOCK >= count > (blocks - 1) * fused_swiglu.BLOCK > 2**31
    assert blocks < 2**31 - 1
    with pytest.raises(ValueError, match='too large'):
        fused_swiglu._grid(2**31 * fused_swiglu.BLOCK)


@pytest.mark.parametrize('strided', [True, False])
def test_bf16_eager_rounding_contract(strided):
    torch.manual_seed(26)
    x = torch.randn(2, 29, 31, dtype=torch.bfloat16).requires_grad_()
    y = torch.randn_like(x).requires_grad_()
    g = torch.randn(2, 1, 31, dtype=x.dtype).expand_as(x) if strided else torch.randn_like(x)
    actual = torch.nn.functional.silu(x) * y
    dx, dy = torch.autograd.grad(actual, (x, y), g)
    # Emulate native CUDA intermediates, including its two BF16 materializations.
    xf, yf, gf = x.detach().float(), y.detach().float(), g.float()
    denom = 1 + (-xf).exp()
    act = (xf / denom).bfloat16().float()
    sigmoid = 1 / denom
    ref_dx = ((gf * yf).bfloat16().float() * sigmoid * (1 + xf * (1 - sigmoid))).bfloat16()
    torch.testing.assert_close(actual, (act * yf).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(dx, ref_dx, rtol=0, atol=0)
    torch.testing.assert_close(dy, (gf * act).bfloat16(), rtol=0, atol=0)
    # A standard FP32 SwiGLU with only a final BF16 cast changes real outputs.
    assert not torch.equal(actual, ((xf / denom) * yf).bfloat16())


def test_projection_wiring_preserves_parameters_and_activations(monkeypatch):
    from types import SimpleNamespace
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
    from peft import LoraConfig, inject_adapter_in_model

    cfg = Qwen3_5TextConfig(hidden_size=8, intermediate_size=16)
    layers = torch.nn.ModuleList(Qwen3_5MLP(cfg, intermediate_size=16) for _ in range(64))
    lora = LoraConfig(r=2, lora_alpha=4, init_lora_weights=False,
                      target_modules=['up_proj', 'down_proj', 'gate_proj'])
    trained = inject_adapter_in_model(lora, layers)
    original = copy.deepcopy(trained[0])
    params_before = dict(trained.named_parameters())
    observed = []

    def reference(gate, up):
        observed.append((gate.shape, up.shape))
        return torch.nn.functional.silu(gate) * up

    model = SimpleNamespace(get_base_model=lambda: SimpleNamespace(model=SimpleNamespace(
        language_model=SimpleNamespace(layers=[SimpleNamespace(mlp=m) for m in trained]))))
    monkeypatch.setattr(fused_swiglu, 'triton', object())
    monkeypatch.setattr(fused_swiglu._FusedSwiGLU, 'apply', reference)
    assert fused_swiglu.install_fused_swiglu(model)['fused_language_mlps'] == 64
    with pytest.raises(ValueError, match='unmodified'):
        fused_swiglu.install_fused_swiglu(model)
    assert params_before.keys() == dict(trained.named_parameters()).keys()
    assert all(p is params_before[name] for name, p in trained.named_parameters())
    x = torch.randn(1, 19, 8, requires_grad=True)
    x0 = x.detach().clone().requires_grad_()
    y = trained[0](x)
    expected = original(x0)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    assert observed == [((1, 19, 16), (1, 19, 16))]
    g = torch.randn_like(y)
    y.backward(g)
    expected.backward(g)
    torch.testing.assert_close(x.grad, x0.grad, rtol=0, atol=0)
    for name, p in original.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(trained[0].get_parameter(name).grad, p.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton activation parity')
@pytest.mark.parametrize('strided', [False, True])
def test_cuda_kernel_parity(strided):
    torch.manual_seed(26)
    x = torch.randn((2, 511, 257), dtype=torch.bfloat16, device='cuda').requires_grad_()
    y = torch.randn_like(x).requires_grad_()
    g = torch.randn_like(x)
    if strided:
        g = g[:, :1, :].expand_as(x)
    actual = fused_swiglu._FusedSwiGLU.apply(x, y)
    expected = torch.nn.functional.silu(x) * y
    grads = torch.autograd.grad(actual, (x, y), g)
    reference = torch.autograd.grad(expected, (x, y), g)
    for result, target in zip((actual, *grads), (expected, *reference), strict=True):
        torch.testing.assert_close(result, target, rtol=0, atol=0)
