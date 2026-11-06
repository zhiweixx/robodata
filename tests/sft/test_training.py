"""Focused experiment-specific regressions. No production weights, datasets, or GPU."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robodata.sft import policy, training
from robodata.sft.run_io import atomic, write_json


def test_exact_epoch_and_warmup():
    rows = [dict(episode_id=f's{i}', action_tokens=int(i < 46)) for i in range(205)]
    groups = policy.make_groups(rows)
    assert groups == policy.make_groups(rows[::-1])
    assert [len(g) for g in groups] == [8] * 25 + [5]
    assert all(sum(r['action_tokens'] for r in g) >= 1 for g in groups)
    assert len({r['episode_id'] for g in groups for r in g}) == 205
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
