"""Focused experiment-specific regressions. No production weights, datasets, or GPU."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robodata.sft import policy, training
from robodata.sft.run_io import atomic, write_json


def test_wandb_logs_validation_once_at_final_optimizer_step(monkeypatch, tmp_path):
    import sys
    from unittest.mock import Mock, call
    from robodata.sft.run_io import OfflineMetrics

    wandb = Mock()
    run = wandb.init.return_value
    run.id, run.dir, run.summary = 'test', str(tmp_path / 'files'), {}
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    rows = [{'optimizer_step': n, 'train/loss': 1 / n} for n in range(1, 27)]
    metrics = OfflineMetrics(tmp_path, {'run_id': 'test'}, rows[:-1])
    metrics.update(rows[-1])
    validation = {'val/episodes': 11, 'val/balanced_accuracy': 0.875}
    summary = {'epoch/updates': 26, **validation}
    metrics.finish(summary)

    assert run.log.call_args_list == [
        *(call(row, step=row['optimizer_step'], commit=False) for row in rows),
        call(validation, step=26, commit=True),
    ]
    assert run.summary == summary


def test_exact_epoch_and_warmup():
    rows = [dict(episode_id=f's{i}', action_tokens=int(i < 46)) for i in range(201)]
    groups = policy.make_groups(rows)
    assert groups == policy.make_groups(rows[::-1])
    assert [len(g) for g in groups] == [8] * 25 + [1]
    assert all(sum(r['action_tokens'] for r in g) >= 1 for g in groups)
    assert len({r['episode_id'] for g in groups for r in g}) == 201
    assert [policy.learning_rate(i, 26) for i in (1, 2, 26)] == [2.5e-5, 5e-5, 5e-6]


def test_manifest_gates_before_loading_any_model(tmp_path):
    data = {'model_revision': policy.BASE_REVISION, 'context_ready': False,
            'over_limit': {'test_s0_l1': 262145}}
    data['fingerprint'] = hashlib.sha256(
        json.dumps(data, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    write_json(tmp_path / 'manifest.json', data)
    with pytest.raises(ValueError, match='test_s0_l1'):
        training._dataset(tmp_path)
    data['over_limit'].clear()
    write_json(tmp_path / 'manifest.json', data)
    with pytest.raises(ValueError, match='fingerprint'):
        training._dataset(tmp_path)


def test_failed_checkpoint_write_preserves_committed_bytes(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'original')
    def interrupted(f):
        f.write(b'partial')
        raise RuntimeError('process interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        atomic(path, interrupted)
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == b'original'


def test_only_the_five_measured_overlength_states_are_removed(tmp_path):
    from robodata.sft.encoding import CONTEXT_LIMIT, EXCLUDED_LENGTHS, select_eligible
    split = json.loads(Path(policy.__file__).with_name('split_manifest.json').read_bytes())['states']
    rows = [dict(episode_id=sid, seq_length=EXCLUDED_LENGTHS.get(sid, 2000),
                 termination_reason='environment_ended', action_tokens=int(s['success']), **s)
            for sid, s in split.items() if s['split'] != 'unscored']
    lengths = {r['episode_id']: r['seq_length'] for r in rows}
    manifest = dict(context_limit=CONTEXT_LIMIT, model_revision=policy.BASE_REVISION, context_ready=False,
                    over_limit=EXCLUDED_LENGTHS, rows=rows, counts={'train': {}, 'heldout': {}})
    manifest['fingerprint'] = hashlib.sha256(
        json.dumps(manifest, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    original = tmp_path / 'manifest.json'
    write_json(original, manifest)
    before = original.read_bytes()
    report = dict(context_limit=CONTEXT_LIMIT, over_limit=EXCLUDED_LENGTHS, lengths=lengths,
                  max_length=max(lengths.values()))
    write_json(tmp_path / 'length_report.json', report)
    selected = select_eligible(tmp_path)
    assert original.read_bytes() == before
    assert selected['full_manifest_sha256'] == hashlib.sha256(before).hexdigest()
    assert selected['rows'] == [r for r in rows if r['episode_id'] not in EXCLUDED_LENGTHS]
    assert selected['counts'] == {'train': {'episodes': 201}, 'heldout': {'episodes': 11}}
    lengths['imitate_sorting_sequence_s0_l0'] -= 1
    write_json(tmp_path / 'length_report.json', report)
    with pytest.raises(ValueError, match='measured length'):
        select_eligible(tmp_path)


def test_validation_uses_seven_failures_and_three_natural_failures():
    rows = []
    for success, reason, count in [(True, 'environment_ended', 4), (False, 'model_give_up', 4),
                                   (False, 'environment_ended', 3)]:
        for _ in range(count):
            rows.append(dict(success=success, termination_reason=reason, pred_success=reason != 'model_give_up',
                             action_tokens=int(success), outcome_tokens=2, action_nll_sum=float(success),
                             outcome_nll_sum=2., binary_nll=1.))
    summary = training._validation_summary(rows)
    assert summary['val/episodes'] == 11 and summary['val/natural_failure_total'] == 3
    assert summary['val/TP'] == 4 and summary['val/TN'] == 4
    assert summary['val/binary_nll'] == 1. and summary['val/balanced_accuracy'] == pytest.approx(0.5 * (1 + 4/7))


def test_sparse_two_episode_backward_matches_dense_token_weighting():
    torch = pytest.importorskip('torch')
    torch.manual_seed(123)
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 7)
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.head = torch.nn.Linear(7, 16, bias=False)
        def get_base_model(self):
            return self
        def get_output_embeddings(self):
            return self.head
    model = Model()
    # 130 supervised tokens exercises rematerialization on both sides of the chunk boundary.
    ids1 = torch.randint(16, (1, 132))
    ids2 = torch.randint(16, (1, 8))
    episodes = [(ids1, torch.arange(2, 132), torch.tensor([1] * 128 + [2, 2])),
                (ids2, torch.tensor([6, 7]), torch.tensor([2, 2]))]
    for ids, pos, roles in episodes:
        a, o, _ = training._sparse_loss(model, {'input_ids': ids}, pos, roles)
        (a / 128 + .25 * o / 4).backward()
    sparse_grad = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    actions, outcomes = [], []
    for ids, pos, roles in episodes:
        logits = model.head(model.model(ids).last_hidden_state)[0, pos - 1]
        nll = torch.nn.functional.cross_entropy(logits, ids[0, pos], reduction='none')
        actions.append(nll[roles == 1])
        outcomes.append(nll[roles == 2])
    (torch.cat(actions).mean() + .25 * torch.cat(outcomes).mean()).backward()
    for sparse, param in zip(sparse_grad, model.parameters(), strict=True):
        torch.testing.assert_close(param.grad, sparse)


def test_verdict_ignores_eos_logits_and_uses_prior_state():
    torch = pytest.importorskip('torch')
    ids = torch.tensor([[12, 3, policy.FAILURE_ID, policy.END_ID]])
    logits = torch.zeros((1, 2, max(policy.SUCCESS_ID, policy.FAILURE_ID) + 1))
    logits[0, 0, policy.SUCCESS_ID] = 2.
    logits[0, 1, policy.FAILURE_ID] = 60.
    pred, nll = policy.verdict(logits, torch.tensor([2, 3]), torch.tensor([2, 2]), ids, False)
    assert pred is True
    assert nll == pytest.approx(2.126928, abs=1e-6)


def test_committed_peft_and_adam_resume_at_exact_next_episode(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    peft = pytest.importorskip('peft')
    from robodata.sft.training import _checkpoint, _restore
    # CPU RNG stand-in exercises serialization without touching CUDA.
    monkeypatch.setattr(torch.cuda, 'get_rng_state', torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', torch.set_rng_state)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
        def forward(self, x):
            return self.proj(x)

    def construct():
        torch.manual_seed(12)
        model = peft.get_peft_model(Toy(), peft.LoraConfig(r=2, lora_alpha=2,
                                                           target_modules=['proj'], lora_dropout=0))
        return model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)

    def update(model, optimizer):
        optimizer.zero_grad()
        out = model(torch.tensor([[.1, -.9]]))
        out.square().sum().backward()
        optimizer.step()
        return out.detach()

    groups = [[{'episode_id': 'episode-a'}], [{'episode_id': 'episode-b'}]]
    identity = {'manifest': 'fixed'}
    history = [{'optimizer_step': 1, 'train/episodes_seen': 1}]
    first, adam = construct()
    update(first, adam)
    _checkpoint(tmp_path, first, adam, 1, groups, history, identity)
    step2 = update(first, adam)
    second, resumed = construct()
    assert _restore(tmp_path, second, resumed, groups, identity) == history
    with pytest.raises(ValueError, match='identity|different'):
        _restore(tmp_path, second, resumed, groups, {'manifest': 'different'})
    torch.testing.assert_close(update(second, resumed), step2)
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p, q)
