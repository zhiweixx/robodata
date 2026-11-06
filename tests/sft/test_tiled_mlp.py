"""CPU comparison against the native Qwen MLP with nonzero LoRA A and B matrices."""
import copy

import pytest


def test_tiled_mlp_preserves_parameters_outputs_and_gradients():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from robodata.sft.tiled_mlp import tile_language_mlp

    torch.manual_seed(14)
    mlp = qwen.Qwen3_5MLP(Qwen3_5TextConfig(hidden_size=32), intermediate_size=48)
    config = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, init_lora_weights=False,
                             target_modules=["gate_proj", "up_proj", "down_proj"])
    native = peft.inject_adapter_in_model(config, mlp)
    tiled = copy.deepcopy(native)
    parameters = dict(tiled.named_parameters())
    tile_language_mlp(tiled, tile_size=7)
    assert list(tiled.state_dict()) == list(native.state_dict())
    assert all(param is parameters[name] for name, param in tiled.named_parameters())

    gate_shapes = []
    tiled.gate_proj.register_forward_pre_hook(
        lambda module, inputs: gate_shapes.append(inputs[0].shape[-2]))

    for track_input_grad in (True, False):
        x = torch.randn(1, 25, 32, requires_grad=track_input_grad)
        xt = x.detach().clone().requires_grad_(track_input_grad)
        # Exercise the same nested nonreentrant checkpoint as a training decoder layer.
        expected = checkpoint(native, x, use_reentrant=False)
        actual = checkpoint(tiled, xt, use_reentrant=False)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream)
        if track_input_grad:
            torch.testing.assert_close(xt.grad, x.grad, rtol=2e-5, atol=1e-6)
        trainable = 0
        for name, param in native.named_parameters():
            other = tiled.get_parameter(name)
            if param.requires_grad:
                trainable += 1
                assert param.grad.abs().sum() > 0
                torch.testing.assert_close(other.grad, param.grad, rtol=2e-5, atol=5e-6)
            else:
                assert other.grad is None and param.grad is None
        assert trainable == 6
        native.zero_grad(set_to_none=True)
        tiled.zero_grad(set_to_none=True)

    assert gate_shapes and set(gate_shapes) == {7}
    gate_shapes.clear()
    with torch.inference_mode():
        torch.testing.assert_close(tiled(x), native(x), rtol=2e-5, atol=1e-6)
    assert gate_shapes == [7, 7, 7, 7]
