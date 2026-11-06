"""Memory-bounded Qwen language FFN; preserves attention and adapter parameter names."""
from functools import wraps

FFN_TILE_SIZE = 1024


def tile_language_mlp(module, tile_size=FFN_TILE_SIZE):
    """Checkpoint each positionwise FFN block to bound both projection and backward memory."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    if type(module) is not Qwen3_5MLP or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("Expected the pinned language MLP and a positive tile size")
    if "forward" in vars(module):
        raise ValueError("Language MLP already has a forward override")
    forward = module.forward

    @wraps(forward)
    def tiled(x):
        if x.shape[-2] <= tile_size:
            return forward(x)
        outputs = []
        for tile in x.split(tile_size, dim=-2):
            if torch.is_grad_enabled():
                # Nonreentrant checkpoint also works when x is frozen and LoRA alone requires grad.
                outputs.append(checkpoint(forward, tile, use_reentrant=False))
            else:
                outputs.append(forward(tile))
        return torch.cat(outputs, dim=-2)

    module.forward = tiled


def tile_qwen_language_ffns(peft_model, tile_size=FFN_TILE_SIZE):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    layers = peft_model.get_base_model().model.language_model.layers
    if len(layers) != 64 or any(type(layer.mlp) is not Qwen3_5MLP for layer in layers):
        raise ValueError("Expected 64 language FFNs of the pinned Qwen model")
    for layer in layers:
        tile_language_mlp(layer.mlp, tile_size)
