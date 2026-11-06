"""One-epoch multimodal LoRA SFT on complete RoboDojo episodes (experiment B)."""
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile
import time
import uuid

from robodata.sft import policy
from robodata.sft.run_io import OfflineMetrics, atomic, locked, read_json, sha256, write_json, write_metrics


def _dataset(prepared):
    manifest = read_json(prepared / 'manifest.json')
    fingerprint = dict(manifest)
    digest = fingerprint.pop('fingerprint')
    serialized = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                            allow_nan=False).encode()
    if hashlib.sha256(serialized).hexdigest() != digest or manifest['model_revision'] != policy.BASE_REVISION:
        raise ValueError('Prepared manifest fingerprint or model revision mismatch')
    if manifest.get('context_ready') is not True or manifest.get('over_limit'):
        raise ValueError(f"Preparation context gate failed: {manifest.get('over_limit')}")
    if manifest['context_limit'] != 262144:
        raise ValueError('Unexpected context limit')
    rows = manifest['rows']
    from robodata.sft.encoding import PROCESSOR_HASHES, TEMPLATE_FLAGS
    split_file = Path(__file__).with_name('split_manifest.json')
    release = read_json(split_file)
    if (manifest['model'] != 'Qwen/Qwen3.8-27B' or manifest['dataset'] != release['dataset']
            or manifest['dataset_revision'] != policy.DATA_REVISION
            or manifest['split_sha256'] != sha256(split_file)
            or manifest['archive_manifest_sha256'] != sha256(prepared / 'archive_manifest.json')
            or manifest['template_flags'] != TEMPLATE_FLAGS or manifest['processor_hashes'] != PROCESSOR_HASHES):
        raise ValueError('Dataset release, split, processor or template fingerprint mismatch')
    pinned = release['states']
    if len(rows) != 217 or len({r['episode_id'] for r in rows}) != 217:
        raise ValueError('Expected 217 unique scored states')
    for row in rows:
        expected = pinned[row['episode_id']]
        if (row['split'] != expected['split'] or row['success'] is not expected['success']
                or row['semantic_state_sha256'] != expected['semantic_state_sha256']):
            raise ValueError(f"Wrong split or outcome for {row['episode_id']}")
        if not 0 < row['seq_length'] <= 262144 or row['outcome_tokens'] != 2:
            raise ValueError(f"Invalid encoded sequence: {row['episode_id']}")
        if row['success'] != bool(row['action_tokens']) or row['success'] != bool(row['action_turns']):
            raise ValueError(f"Incorrect action supervision: {row['episode_id']}")
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'heldout']
    counts = tuple((len(s), sum(r['success'] for r in s), sum(r['action_turns'] for r in s))
                   for s in (train, val))
    if counts != ((205, 46, 603), (12, 4, 83)):
        raise ValueError(f'Unexpected split counts: {counts}')
    return manifest, train, val


def _identity(prepared, model_dir, fingerprint):
    pinned = read_json(Path(__file__).with_name('weight_hashes.json'))
    index_path = model_dir / 'model.safetensors.index.json'
    if sha256(index_path) != pinned['index_sha256'] or pinned['revision'] != policy.BASE_REVISION:
        raise ValueError('Model index differs from pinned base')
    if set(read_json(index_path)['weight_map'].values()) != set(pinned['shards']):
        raise ValueError('Unexpected weight shard set')
    for name, expected in pinned['shards'].items():
        if sha256(model_dir / name) != expected:
            raise ValueError(f'Base-model weight hash differs: {name}')
    code = {n: sha256(Path(__file__).with_name(n)) for n in
            ('policy.py', 'training.py', 'run_io.py', 'encoding.py', 'weight_hashes.json', 'split_manifest.json')}
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'wandb', 'fla-core', 'accelerate')}
    return {
        'manifest_sha256': sha256(prepared / 'manifest.json'), 'data_fingerprint': fingerprint,
        'model_index_sha256': pinned['index_sha256'],
        'base_revision': policy.BASE_REVISION, 'dataset_revision': policy.DATA_REVISION,
        'code_sha256': code, 'versions': versions, 'seed': policy.SEED,
        'rank': 16, 'alpha': 32, 'dropout': 0, 'batch_size': 8,
        'outcome_weight': 0.25, 'epoch_count': 1, 'max_lr': 5e-5, 'min_lr': 5e-6,
        'warmup_updates': 2, 'weight_decay': 0, 'gradient_clip': 1.0,
    }


def _base_model(model_dir):
    import torch
    from transformers import AutoModelForImageTextToText
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This experiment requires one visible H200')
    # Prevent quadratic eager fallback in full-length SDPA.
    # Transformers 5.19 selects the installed FLA package before its reference loop;
    # do not download hub kernels. An absent FLA package makes full-episode training impractical.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    if not callable(chunk_gated_delta_rule):
        raise RuntimeError('Install the pinned local FLA training kernel')
    torch.backends.cuda.enable_math_sdp(False)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, use_kernels=False, device_map={'': 0})
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model


def _new_model(model_dir, run):
    import torch
    from peft import LoraConfig, get_peft_model
    base = _base_model(model_dir)
    targets = policy.lora_targets(base)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                            bias='none', target_modules=targets))
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for _, p in params)
    if count != policy.EXPECTED_PARAMS or len(params) != 2 * len(targets):
        raise ValueError(f'Incorrect adapter topology: {len(params)} tensors / {count} parameters')
    for name, param in params:
        if '.lora_A.' not in name and '.lora_B.' not in name:
            raise ValueError(f'Unexpected trainable parameter: {name}')
        if param.dtype != torch.float32:
            raise ValueError('Expected FP32 trainable adapters with frozen BF16 base')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    write_json(run / 'lora_targets.json', {'targets': targets, 'trainable_parameters': count})
    print(f'Verified {len(targets)} language projections and {count:,} LoRA parameters', flush=True)
    return model


def _inputs(prepared, row, processor):
    from robodata.sft.encoding import load_episode
    import torch
    kwargs, positions, roles = load_episode(prepared, row, processor)
    if kwargs['input_ids'].shape != (1, row['seq_length']):
        raise ValueError('Rehydrated episode length differs from manifest')
    if positions.ndim != 1 or len(positions) < 2 or positions.dtype != torch.long or positions.shape != roles.shape:
        raise ValueError('Invalid gold-position/role tensors')
    if not (positions[0] > 0 and positions[-1] < row['seq_length'] and (positions[1:] > positions[:-1]).all()):
        raise ValueError('Invalid causal target positions')
    if ((roles == 1).sum().item(), (roles == 2).sum().item(), len(roles)) != (
            row['action_tokens'], row['outcome_tokens'], row['action_tokens'] + row['outcome_tokens']):
        raise ValueError('Encoded targets differ from optimizer denominators')
    kwargs = {k: t.to('cuda', dtype=torch.bfloat16 if t.is_floating_point() else t.dtype)
              for k, t in kwargs.items()}
    return kwargs, positions.cuda(), roles.cuda()


def _project_loss(head, states, gold, roles):
    import torch.nn.functional as F
    logits = head(states).float()
    per_target = F.cross_entropy(logits, gold, reduction='none')
    return per_target[roles == 1].sum(), per_target[roles == 2].sum()


def _sparse_loss(model, kwargs, positions, roles, *, classify=None):
    """Gather predecessor states and project only 128 targets at a time."""
    import torch
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    head = base.get_output_embeddings()
    hidden = base.model(**kwargs, use_cache=False, return_dict=True).last_hidden_state[0]
    selected = hidden.index_select(0, positions - 1)
    gold = kwargs['input_ids'][0, positions]
    sums = [hidden.new_zeros((), dtype=torch.float32), hidden.new_zeros((), dtype=torch.float32)]
    for chunk, labels, kinds in zip(selected.split(128), gold.split(128), roles.split(128), strict=True):
        if torch.is_grad_enabled():
            a, o = checkpoint(_project_loss, head, chunk, labels, kinds, use_reentrant=False)
        else:
            a, o = _project_loss(head, chunk, labels, kinds)
        sums[0] = sums[0] + a
        sums[1] = sums[1] + o
    decision = None
    if classify is not None:
        # Reuse backbone states; only the outcome token and its EOS are projected here.
        sel = roles == 2
        decision = policy.verdict(head(selected[sel])[None], positions[sel], roles[sel],
                                  kwargs['input_ids'], classify)
    return sums[0], sums[1], decision


def _offload_activations(model):
    import torch
    # Shared parameter storages are already resident on GPU. Do not offload copies
    # of their views (e.g. a transposed LM head) for every checkpointed layer.
    model_storages = {v.untyped_storage().data_ptr() for v in model.parameters()}
    def pack(tensor):
        tensor = tensor.detach()
        if not tensor.is_cuda or tensor.untyped_storage().data_ptr() in model_storages:
            return tensor
        return (tensor.device, tensor.cpu())
    def unpack(value):
        return value[1].to(value[0]) if isinstance(value, tuple) else value
    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _episode_backward(model, prepared, processor, row, denominators, offload):
    import torch
    kwargs, positions, roles = _inputs(prepared, row, processor)
    context = _offload_activations(model) if offload else contextlib.nullcontext()
    with context, torch.autocast('cuda', dtype=torch.bfloat16):
        a, o, _ = _sparse_loss(model, kwargs, positions, roles)
        loss = a / denominators[0] + policy.OUTCOME_WEIGHT * o / denominators[1]
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"Nonfinite objective on {row['episode_id']}")
    loss.backward()
    return a.item(), o.item()


def _sdpa_check(model, prepared, processor, row):
    import torch
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        kwargs, pos, roles = _inputs(prepared, row, processor)
        sparse_a, sparse_o, _ = _sparse_loss(model, kwargs, pos, roles)
        dense = model(**kwargs, logits_to_keep=0, use_cache=False, return_dict=True).logits
        a, o = policy.target_loss(dense.index_select(1, pos - 1), kwargs['input_ids'], pos, roles)
        for v, ref, n in [(sparse_a, a, row['action_tokens']), (sparse_o, o, row['outcome_tokens'])]:
            torch.testing.assert_close(v / max(n, 1), ref / max(n, 1), rtol=3e-3, atol=0.02)
        return {'episode_id': row['episode_id'], 'seq_length': row['seq_length'],
                'sparse_action_sum': sparse_a.item(), 'dense_action_sum': a.item(),
                'sparse_outcome_sum': sparse_o.item(), 'dense_outcome_sum': o.item()}


def _clear():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _check_gpu(model, prepared, rows, processor, run, identity):
    import torch
    receipt = run / 'preflight.json'
    if receipt.exists():
        old = read_json(receipt)
        if old['identity'] != identity:
            raise ValueError('Preflight identity mismatch')
        if old['passed']:
            return old['offload']
    short = min((r for r in rows if r['action_tokens']), key=lambda r: r['seq_length'])
    longest = max(rows, key=lambda r: r['seq_length'])
    result = {'identity': identity, 'passed': False, 'longest_episode': longest['episode_id'],
              'longest_seq_length': longest['seq_length'], 'attempts': []}
    try:
        result['short_comparison'] = _sdpa_check(model, prepared, processor, short)
        model.train()
        for offload in (False, True):
            _clear()
            torch.cuda.reset_peak_memory_stats()
            ok = False
            try:
                _episode_backward(model, prepared, processor, longest, (
                    max(1, longest['action_tokens']), longest['outcome_tokens']), offload)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               1, error_if_nonfinite=True)
                ok = True
            except torch.cuda.OutOfMemoryError:
                if offload:
                    raise
            finally:
                model.zero_grad(set_to_none=True)
                result['attempts'].append({'offload': offload, 'passed': ok,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
            if ok:
                result.update(passed=True, offload=offload)
                return offload
    finally:
        write_json(receipt, result)
        _clear()


def _initialize_seeds():
    import numpy as np
    import torch
    np.random.seed(policy.SEED)
    random.seed(policy.SEED)
    torch.manual_seed(policy.SEED)
    torch.cuda.manual_seed_all(policy.SEED)


def _checkpoint(run, model, optimizer, step, groups, metrics, identity):
    import torch
    from peft import get_peft_model_state_dict
    import numpy as np
    alg, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        'identity': identity, 'step': step, 'schedule': {'step': step, 'total': len(groups)},
        'next_batch': [r['episode_id'] for r in groups[step]] if step < len(groups) else [],
        'adapter': {n: p.detach().cpu() for n, p in get_peft_model_state_dict(
            model, save_embedding_layers=False).items()},
        'optimizer': optimizer.state_dict(),
        'numpy_rng': (alg, keys.tolist(), pos, has_gauss, cached),
        'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(), 'metrics': metrics,
    }
    atomic(run / 'checkpoint.pt', lambda f: torch.save(state, f))


def _restore(run, model, optimizer, groups, identity):
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    state = torch.load(run / 'checkpoint.pt', weights_only=True, map_location='cpu')
    step = state['step']
    if not 0 < step <= len(groups) or state['identity'] != identity or state['schedule'] != {
            'step': step, 'total': len(groups)}:
        raise ValueError('Checkpoint belongs to a different run or optimizer schedule')
    expected_next = [r['episode_id'] for r in groups[step]] if step < len(groups) else []
    metrics = state['metrics']
    if (state['next_batch'] != expected_next or [r['optimizer_step'] for r in metrics] != list(range(1, step + 1))
            or metrics[-1]['train/episodes_seen'] != sum(len(g) for g in groups[:step])):
        raise ValueError('Checkpoint episode cursor is inconsistent')
    loaded = set_peft_model_state_dict(model, state['adapter'])
    if loaded.unexpected_keys or any('.lora_' in k for k in loaded.missing_keys):
        raise ValueError('Saved trainable parameters differ from the loaded adapter')
    optimizer.load_state_dict(state['optimizer'])
    alg, keys, pos, has_gauss, cached = state['numpy_rng']
    np.random.set_state((alg, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    return metrics


def _update(model, processor, prepared, optimizer, params, group, step, total_steps, seen, offload):
    import torch
    a_tokens = sum(r['action_tokens'] for r in group)
    o_tokens = sum(r['outcome_tokens'] for r in group)
    lr = policy.learning_rate(step, total_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    action_sum = outcome_sum = 0.
    for row in group:
        a, o = _episode_backward(model, prepared, processor, row, (a_tokens, o_tokens), offload)
        action_sum += a
        outcome_sum += o
    grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    tokens = sum(r['seq_length'] for r in group)
    return {
        'optimizer_step': step, 'train/action_nll': action_sum / a_tokens,
        'train/outcome_nll': outcome_sum / o_tokens,
        'train/loss': action_sum / a_tokens + policy.OUTCOME_WEIGHT * outcome_sum / o_tokens,
        'train/action_nll_sum': action_sum, 'train/outcome_nll_sum': outcome_sum,
        'train/action_tokens': a_tokens, 'train/outcome_tokens': o_tokens,
        'train/action_turns': sum(r['action_turns'] for r in group), 'train/outcome_turns': len(group),
        'train/episodes_seen': seen + len(group), 'train/lr': lr, 'train/grad_norm_pre_clip': grad,
        'perf/sequence_tokens': tokens, 'perf/tokens_per_second': tokens / (time.monotonic() - started),
        'perf/peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
        'perf/peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
    }


def _epoch(metrics):
    def total(k):
        return sum(r['train/' + k] for r in metrics)
    if len(metrics) != 26 or (metrics[-1]['train/episodes_seen'], total('action_turns'), total('outcome_turns')) != (
            205, 603, 205):
        raise ValueError('Training epoch is incomplete')
    a, o = total('action_nll_sum') / total('action_tokens'), total('outcome_nll_sum') / total('outcome_tokens')
    return {'epoch/action_nll': a, 'epoch/outcome_nll': o, 'epoch/loss': a + policy.OUTCOME_WEIGHT * o,
            'epoch/episodes': 205, 'epoch/action_turns': 603, 'epoch/outcome_turns': 205,
            'epoch/action_tokens': total('action_tokens'), 'epoch/outcome_tokens': total('outcome_tokens'),
            'epoch/updates': len(metrics)}


def _save_final(run, model, identity):
    dest = run / 'adapter'
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=run, prefix='.final-') as tmp:
            staging = Path(tmp) / 'adapter'
            model.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
            files = {p.name: sha256(p) for p in staging.iterdir() if p.is_file()}
            if 'adapter_model.safetensors' not in files or 'adapter_config.json' not in files:
                raise ValueError('Final adapter serialization incomplete')
            write_json(staging / 'evidence.json', {'identity': identity, 'step': 26, 'files': files})
            os.rename(staging, dest)
    return _verify_adapter(run, identity)


def _verify_adapter(run, identity):
    dest = run / 'adapter'
    info = read_json(dest / 'evidence.json')
    if info['identity'] != identity or info['step'] != 26:
        raise ValueError('Final adapter identity mismatch')
    if any(sha256(dest / f) != digest for f, digest in info['files'].items()):
        raise ValueError('Final adapter checksum mismatch')
    return info['files']['adapter_model.safetensors']


def _evaluate(prepared, model_dir, run, rows, processor, identity):
    import torch
    from peft import PeftModel
    digest = _verify_adapter(run, identity)
    val_path = run / 'validation.json'
    if val_path.exists():
        val = read_json(val_path)
        if val['identity'] != identity or val['adapter_sha256'] != digest:
            raise ValueError('Held-out evaluation identity mismatch')
        return val['summary']
    model = PeftModel.from_pretrained(_base_model(model_dir), run / 'adapter', is_trainable=False)
    model.eval()
    decisions = []
    # A committed row lets an interrupted offline validation continue once per completed state.
    cache = run / 'validation_rows'
    for row in rows:
        path = cache / (row['episode_id'] + '.json')
        if path.exists():
            item = read_json(path)
            if (item['adapter_sha256'], item['episode_id'], item['fingerprint']) != (
                    digest, row['episode_id'], identity['data_fingerprint']):
                raise ValueError('Cached validation row identity mismatch')
        else:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                kwargs, pos, roles = _inputs(prepared, row, processor)
                a, o, (pred, nll) = _sparse_loss(model, kwargs, pos, roles, classify=row['success'])
                item = {'episode_id': row['episode_id'], 'adapter_sha256': digest,
                        'fingerprint': identity['data_fingerprint'], 'success': row['success'],
                        'pred_success': pred, 'termination_reason': row['termination_reason'],
                        'action_nll_sum': a.item(), 'outcome_nll_sum': o.item(), 'binary_nll': nll,
                        'action_tokens': row['action_tokens'], 'outcome_tokens': row['outcome_tokens']}
                del kwargs, pos, roles, a, o
                write_json(path, item)
        decisions.append(item)
    confusion = {k: 0 for k in ('TP', 'FN', 'TN', 'FP')}
    subsets = {'success': [0, 0], 'give_up_failure': [0, 0], 'natural_failure': [0, 0]}
    for item in decisions:
        truth, predicted = item['success'], item['pred_success']
        confusion[{(True, True): 'TP', (True, False): 'FN', (False, False): 'TN',
                   (False, True): 'FP'}[truth, predicted]] += 1
        subset = 'success' if truth else ('give_up_failure' if item['termination_reason'] == 'model_give_up'
                                           else 'natural_failure')
        subsets[subset][0] += int(truth == predicted)
        subsets[subset][1] += 1
    if any(x[1] != 4 for x in subsets.values()):
        raise ValueError(f'Unexpected held-out subtype counts: {subsets}')
    summary = {f'val/{k}': v for k, v in confusion.items()}
    for name, (correct, count) in subsets.items():
        summary['val/' + name + '_correct'] = correct
        summary['val/' + name + '_total'] = count
    def total(k):
        return sum(item[k] for item in decisions)
    summary.update({'val/action_nll': total('action_nll_sum') / total('action_tokens'),
                    'val/outcome_nll': total('outcome_nll_sum') / total('outcome_tokens'),
                    'val/binary_nll': total('binary_nll') / 12,
                    'val/balanced_accuracy': 0.5 * (confusion['TP'] / 4 + confusion['TN'] / 8),
                    'val/episodes': 12, 'val/action_turns': 83, 'val/outcome_turns': 12})
    write_json(val_path, {'identity': identity, 'adapter_sha256': digest, 'summary': summary,
                          'episodes': decisions})
    return summary


def _prepare(prepared_dir, model_path, run_dir):
    from robodata.sft.encoding import load_processor
    prepared, model, run = map(lambda p: Path(p).resolve(), (prepared_dir, model_path, run_dir))
    manifest, rows, val = _dataset(prepared)
    ident = _identity(prepared, model, manifest['fingerprint'])
    return prepared, model, run, load_processor(model), rows, val, ident


def preflight(prepared_dir, model_path, run_dir):
    """Validate exact sparse loss and full longest-episode backward without an optimizer."""
    prep, path, run, processor, rows, _, ident = _prepare(prepared_dir, model_path, run_dir)
    with locked(run):
        _initialize_seeds()
        model = _new_model(path, run)
        _check_gpu(model, prep, rows, processor, run, ident)
        return read_json(run / 'preflight.json')


def train(prepared_dir, model_path, run_dir, *, resume=False):
    import torch
    prep, path, run, processor, rows, val, ident = _prepare(prepared_dir, model_path, run_dir)
    with locked(run):
        config_path = run / 'run.json'
        if config_path.exists():
            cfg = read_json(config_path)
            if cfg['identity'] != ident or not resume:
                raise ValueError('Existing training directory: require matching identity and --resume')
        else:
            if (run / 'checkpoint.pt').exists():
                raise ValueError('Training checkpoint lacks run identity')
            cfg = {'run_id': uuid.uuid4().hex, 'identity': ident}
            write_json(config_path, cfg)
        _initialize_seeds()
        groups = policy.make_groups(rows)
        write_json(run / 'batches.json', [[r['episode_id'] for r in g] for g in groups])
        model = _new_model(path, run)
        offload = _check_gpu(model, prep, rows, processor, run, ident)
        _initialize_seeds()
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=5e-5, betas=(0.9, 0.999), weight_decay=0,
                                     foreach=False)
        history = _restore(run, model, optimizer, groups, ident) if (run / 'checkpoint.pt').exists() else []
        write_metrics(run, history)
        metrics = OfflineMetrics(run, dict(ident, run_id=cfg['run_id']), history)
        try:
            for step in range(len(history) + 1, len(groups) + 1):
                seen = history[-1]['train/episodes_seen'] if history else 0
                update = _update(model, processor, prep, optimizer, params, groups[step-1], step,
                                 len(groups), seen, offload)
                history.append(update)
                _checkpoint(run, model, optimizer, step, groups, history, ident)
                write_metrics(run, history)
                metrics.update(update)
                print(f'Committed update {step}/26 ({update["train/episodes_seen"]}/205 episodes)', flush=True)
            epoch = _epoch(history)
            write_json(run / 'epoch.json', epoch)
            _save_final(run, model, ident)
            del model, params, optimizer
            _clear()
            validation = _evaluate(prep, path, run, val, processor, ident)
        except BaseException:
            metrics.finish()
            raise
        metrics.finish(epoch | validation)
        return epoch | validation


def evaluate(prepared_dir, model_path, run_dir):
    """Evaluate the final adapter, reusing only committed scores of this identical adapter."""
    prep, path, run, processor, _, val, ident = _prepare(prepared_dir, model_path, run_dir)
    with locked(run):
        cfg = read_json(run / 'run.json')
        if cfg['identity'] != ident:
            raise ValueError('Training run identity mismatch')
        summary = _evaluate(prep, path, run, val, processor, ident)
        if not (run / 'wandb_offline.json').exists():
            history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            metrics = OfflineMetrics(run, dict(ident, run_id=cfg['run_id']), history)
            metrics.finish(_epoch(history) | summary)
        return summary
