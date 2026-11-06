"""Keep immutable Qwen language MLP weights in RAM and execute native-shaped GPU linears."""
from functools import wraps

N_LAYERS = 64
N_PROJECTIONS = 3 * N_LAYERS
EXPECTED_BYTES = N_PROJECTIONS * 5120 * 17408 * 2


def offload_frozen_projection(linear):
    """Leave the registered parameter on CPU, with only temporary copies for GEMM."""
    import torch
    if (type(linear) is not torch.nn.Linear or linear.bias is not None or linear.weight.requires_grad
            or linear.weight.dtype != torch.bfloat16 or not linear.weight.is_contiguous()
            or 'forward' in vars(linear)):
        raise ValueError('Expected an unmodified frozen, bias-free BF16 base linear')
    linear.to(device='cpu')

    @wraps(linear.forward)
    def forward(x):
        host = linear.weight.detach()
        if host.device.type != 'cpu' or linear.weight.requires_grad or x.dtype != host.dtype:
            raise ValueError('Expected immutable host BF16 weights and BF16 inputs')
        device = x.device
        weight = host.to(device=device, copy=True, memory_format=torch.contiguous_format)
        if not torch.is_grad_enabled() or not x.requires_grad:
            return torch.nn.functional.linear(x, weight)

        ptr = weight.untyped_storage().data_ptr()
        version = host._version

        def pack(tensor):
            # Native linear saves a view of its frozen matrix for the gradient of x.
            # The autograd graph needs only the layout and immutable CPU parameter.
            if tensor.untyped_storage().data_ptr() != ptr or tensor.dtype != host.dtype:
                raise RuntimeError('Unexpected tensor saved by frozen native linear')
            return tensor.size(), tensor.stride(), tensor.storage_offset()

        def unpack(view):
            if host._version != version:
                raise RuntimeError('Frozen host weight changed before backward')
            restored = host.to(device=device, copy=True, memory_format=torch.contiguous_format)
            return restored.as_strided(*view)

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return torch.nn.functional.linear(x, weight)

    linear.forward = forward


def _shared_views(model, projections):
    """Find retained parameters/buffers sharing a flat allocation with selected weights."""
    from itertools import chain
    # Storage addresses are evaluated while all selected parameters still reside on the GPU.
    # Integer IDs and byte spans do not hold any of their allocations alive.
    selected = {id(m.weight) for m in projections}
    spans = {}
    for module in projections:
        w = module.weight
        storage = w.untyped_storage().data_ptr()
        start = w.storage_offset() * w.element_size()
        spans.setdefault(storage, []).append((start, start + w.nbytes))
    for regions in spans.values():
        regions.sort()
        if any(b > c for (_, b), (c, _) in zip(regions, regions[1:])):
            raise ValueError('Offload targets have overlapping weight storage')
    retained = {}
    for v in chain(model.parameters(), model.buffers()):
        if not v.is_cuda or id(v) in selected or not v.numel():
            continue
        storage = v.untyped_storage().data_ptr()
        if storage not in spans:
            continue
        start = v.storage_offset() * v.element_size()
        # Account for strides: a tied transposed view can cover the same storage.
        stop = start + (1 + sum((size - 1) * stride for size, stride in zip(v.size(), v.stride()))) * v.element_size()
        if any(start < b and a < stop for a, b in spans[storage]):
            raise ValueError('A retained parameter or buffer aliases an offload target')
        retained.setdefault(storage, []).append((start, stop, v))
    return retained


def _release_flat_backing(retained):
    """Re-home overlapping retained views together so a flat slab can actually be freed."""
    import torch
    for views in retained.values():
        views.sort(key=lambda item: item[0])
        i = 0
        while i < len(views):
            j = i + 1
            start, end, first = views[i]
            while j < len(views) and views[j][0] < end:
                end = max(end, views[j][1])
                j += 1
            # Align the new base for all original element types in this alias group.
            start -= start % 16
            byte_view = torch.empty(0, dtype=torch.uint8, device=first.device).set_(
                first.untyped_storage(), start, (end - start,), (1,))
            local = byte_view.clone().untyped_storage()
            for offset, _, view in views[i:j]:
                replacement = torch.empty(0, dtype=view.dtype, device=view.device).set_(
                    local, (offset - start) // view.element_size(), view.size(), view.stride())
                view.data = replacement  # preserve the Parameter object and all genuine ties
            i = j


def offload_native_qwen_mlps(peft_model):
    """Stage the 192 frozen MLP projections; LoRA and other model layers stay on GPU."""
    import torch

    layers = peft_model.get_base_model().model.language_model.layers
    if len(layers) != N_LAYERS:
        raise ValueError('Expected the 64 language layers of the pinned Qwen model')
    targets = []
    for i, layer in enumerate(layers):
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            proj = getattr(layer.mlp, name)
            linear = proj.get_base_layer()
            expected_shape = (5120, 17408) if name == 'down_proj' else (17408, 5120)
            if (tuple(linear.weight.shape) != expected_shape or not linear.weight.is_cuda
                    or not linear.weight.is_contiguous() or linear.weight.requires_grad
                    or linear.weight.dtype != torch.bfloat16
                    or any(not p.is_cuda for p in proj.lora_A.parameters())
                    or any(not p.is_cuda for p in proj.lora_B.parameters())):
                raise ValueError(f'Unexpected weight/adapter topology on layer {i} {name}')
            targets.append(linear)
    # A tied Parameter exposed outside these target modules cannot be moved independently.
    target_ids = {id(m.weight) for m in targets}
    if len(target_ids) != N_PROJECTIONS or sum(id(p) in target_ids for _, p in
            peft_model.named_parameters(remove_duplicate=False)) != N_PROJECTIONS:
        raise ValueError('Offload target shares a Parameter object')
    retained = _shared_views(peft_model, targets)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for linear in targets:
        offload_frozen_projection(linear)
    _release_flat_backing(retained)
    torch.cuda.synchronize()
    freed = before - torch.cuda.memory_allocated()
    host_bytes = sum(m.weight.nbytes for m in targets if m.weight.device.type == 'cpu')
    print(f'Placed {host_bytes / 2**30:.3f} GiB of frozen MLP weights in RAM; '
          f'live CUDA allocations fell by {freed / 2**30:.3f} GiB', flush=True)
    if host_bytes != EXPECTED_BYTES or freed < EXPECTED_BYTES - 16 * 2**20:
        raise RuntimeError(f'Frozen weight residency check failed: {host_bytes} host bytes, '
                           f'{freed} live GPU bytes released, expected {EXPECTED_BYTES}')
    return {'selected_mlp_layers': N_LAYERS, 'offloaded_frozen_projections': N_PROJECTIONS,
            'offloaded_weight_bytes': host_bytes, 'decrease_in_live_cuda_bytes': freed,
            'shared_cuda_slabs_released': len(retained)}
