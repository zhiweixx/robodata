"""CPU checks for native-shape BF16 weight staging with trainable FP32 LoRA."""
import copy
import weakref

import pytest


@pytest.mark.parametrize("checkpointed", [True, False])
@pytest.mark.parametrize("input_grad", [True, False])
def test_native_weights_forward_and_all_adapters(checkpointed, input_grad):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from robodata.sft.native_weight_offload import offload_frozen_projection

    torch.manual_seed(231)
    mlp = qwen.Qwen3_5MLP(Qwen3_5TextConfig(hidden_size=64), intermediate_size=128).bfloat16()
    config = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, init_lora_weights=False,
                             target_modules=["up_proj", "gate_proj", "down_proj"])
    resident = peft.inject_adapter_in_model(config, mlp)
    for param in resident.parameters():
        if param.requires_grad:
            param.data = param.float()
    staged = copy.deepcopy(resident)
    adapters = {name: p for name, p in staged.named_parameters() if p.requires_grad}
    for name in ("gate_proj", "up_proj", "down_proj"):
        offload_frozen_projection(getattr(staged, name).get_base_layer())
    assert len(adapters) == 6
    assert all(staged.get_parameter(name) is p for name, p in adapters.items())

    x = torch.randn(1, 43, 64, dtype=torch.bfloat16).requires_grad_(input_grad)
    xs = x.detach().clone().requires_grad_(input_grad)
    with torch.autocast("cpu", torch.bfloat16):
        expected = checkpoint(resident, x, use_reentrant=False) if checkpointed else resident(x)
        # Exercise the same nesting as the outer activation offload and inner layer checkpoint.
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
            actual = checkpoint(staged, xs, use_reentrant=False) if checkpointed else staged(xs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    expected.backward(upstream)
    actual.backward(upstream)
    if input_grad:
        torch.testing.assert_close(xs.grad, x.grad, rtol=0, atol=0)
    assert list(staged.state_dict()) == list(resident.state_dict())
    for name, param in resident.named_parameters():
        other = staged.get_parameter(name)
        torch.testing.assert_close(other, param, rtol=0, atol=0)
        if param.requires_grad:
            assert param.grad.abs().sum() > 0
            torch.testing.assert_close(other.grad, param.grad, rtol=0, atol=0)
        else:
            assert param.grad is None and other.grad is None
    with torch.inference_mode(), torch.autocast("cpu", torch.bfloat16):
        torch.testing.assert_close(staged(xs), resident(x), rtol=0, atol=0)


def test_staged_weight_is_released_before_backward(monkeypatch):
    torch = pytest.importorskip("torch")
    from robodata.sft.native_weight_offload import offload_frozen_projection

    layer = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16).requires_grad_(False)
    offload_frozen_projection(layer)
    native = torch.nn.functional.linear
    copies = []

    def observe(x, weight):
        assert weight.untyped_storage().data_ptr() != layer.weight.untyped_storage().data_ptr()
        copies.append(weakref.ref(weight))
        return native(x, weight)

    x = torch.randn(2, 18, 16, dtype=torch.bfloat16, requires_grad=True)
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.functional, "linear", observe)
        output = layer(x)
    assert len(copies) == 1 and copies[0]() is None
    output.sum().backward()
    assert x.grad is not None and layer.weight.grad is None


def test_compacting_flat_weights_preserves_tied_views_and_releases_unused_spans():
    torch = pytest.importorskip("torch")
    from robodata.sft.native_weight_offload import _release_flat_backing

    flat = torch.arange(128, dtype=torch.float32)
    matrix = torch.nn.Parameter(flat[12:24].reshape(3, 4), requires_grad=False)
    tied = matrix.detach().T
    separate = torch.nn.Parameter(flat[64:72], requires_grad=False)
    old_ptr = flat.untyped_storage().data_ptr()
    size = flat.element_size()
    expected = [v.clone() for v in (matrix, tied, separate)]
    # Both differently strided matrices cover the exact same bytes.
    _release_flat_backing({old_ptr: [(64 * size, 72 * size, separate),
                                     (12 * size, 24 * size, matrix),
                                     (12 * size, 24 * size, tied)]})
    for view, original in zip((matrix, tied, separate), expected):
        torch.testing.assert_close(view, original, rtol=0, atol=0)
    shared = matrix.untyped_storage()
    standalone = separate.untyped_storage()
    assert old_ptr != shared.data_ptr() == tied.untyped_storage().data_ptr()
    assert standalone.data_ptr() not in (old_ptr, shared.data_ptr())
    assert shared.nbytes() == 12 * size and standalone.nbytes() == 8 * size
    assert tied.stride() == (1, 4)
    matrix[1, 3] = -10
    assert tied[3, 1] == -10 and flat[19] == 19


def test_short_gate_accepts_equal_losses_and_gradients(monkeypatch):
    torch = pytest.importorskip("torch")
    from robodata.sft import training

    model = torch.nn.Linear(2, 2, bias=False)
    row = {"episode_id": "short", "action_tokens": 5, "outcome_tokens": 2, "seq_length": 8}

    def equal_pass(model, *_args):
        model.weight.grad = torch.ones_like(model.weight)
        return 4.0, 2.0

    monkeypatch.setattr(training, "_episode_backward", equal_pass)
    monkeypatch.setattr(training, "offload_native_qwen_mlps", lambda _: {"decrease_in_live_cuda_bytes": 0})
    result = training._short_ffn_check(model, None, None, row)
    assert result["passed"] and result["gradient_relative_l2"] == 0
