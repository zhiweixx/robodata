"""Batch construction and loss accounting for the full-episode B experiment."""
import math
import random
import re

SEED = 20261106
BASE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
DATA_REVISION = '7df93dc28f4e0495812bad917eed6432360f960d'
BATCH = 8
OUTCOME_WEIGHT = 0.25
SUCCESS_ID, FAILURE_ID, END_ID = 5453, 27997, 248046
TARGETS_RE = (
    r'model\.language_model\.layers\.\d+\.(?:'
    r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|'
    r'linear_attn\.(?:in_proj_qkv|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))'
)
EXPECTED_MODULES = 352
EXPECTED_PARAMS = 100_139_008


def lora_targets(model):
    from torch.nn import Linear
    matches = [name for name, m in model.named_modules()
               if isinstance(m, Linear) and re.fullmatch(TARGETS_RE, name)]
    if len(matches) != EXPECTED_MODULES:
        raise ValueError(f'Expected {EXPECTED_MODULES} scoped language projections, got {len(matches)}')
    return matches


def make_groups(rows, seed=SEED, size=BATCH):
    """Assign one action-bearing episode per group, then shuffle remaining slots."""
    if size <= 0 or not rows:
        raise ValueError('Training requires a positive batch size and at least one episode')
    ordered = sorted(rows, key=lambda row: row['episode_id'])
    if len({row['episode_id'] for row in ordered}) != len(ordered):
        raise ValueError('Repeated episode in epoch')
    positives = [row for row in ordered if row['action_tokens'] > 0]
    remaining = [row for row in ordered if row['action_tokens'] == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    n = math.ceil(len(rows) / size)
    if len(positives) < n:
        raise ValueError('Insufficient action-bearing episodes for the fixed batch schedule')
    groups = [[row] for row in positives[:n]]
    remaining += positives[n:]
    rng.shuffle(remaining)
    for i, group in enumerate(groups):
        slots = min(size, len(rows) - i * size) - 1
        group.extend(remaining[:slots])
        del remaining[:slots]
        rng.shuffle(group)
    return groups


def learning_rate(step, total_steps):
    """One-based update index: two warmup steps, then cosine to exactly 5e-6."""
    if not 1 <= step <= total_steps or total_steps <= 2:
        raise ValueError('Invalid optimizer step')
    if step <= 2:
        return 5e-5 * step / 2
    frac = (step - 2) / (total_steps - 2)
    return 5e-6 + (5e-5 - 5e-6) * (1 + math.cos(math.pi * frac)) / 2


def target_loss(logits, input_ids, positions, roles):
    """NLL sums for the two tasks. logits must already be projected at positions-1."""
    import torch
    from torch.nn.functional import cross_entropy
    if logits.ndim != 3 or logits.shape[:2] != (1, len(positions)):
        raise ValueError('Expected one row of logits per supervised position')
    gold = input_ids[0].index_select(0, positions)
    nll = cross_entropy(logits[0].float(), gold, reduction='none')
    return nll[roles == 1].sum(), nll[roles == 2].sum()


def verdict(logits, positions, roles, input_ids, expected):
    """Forced choice from the pre-answer logits. EOS is excluded from binary NLL."""
    import torch
    slots = (roles == 2).nonzero().flatten()
    if len(slots) != 2:
        raise ValueError('Verdict must supervise one answer token and its end marker')
    expected_id = SUCCESS_ID if expected else FAILURE_ID
    gold = input_ids[0, positions[slots]].tolist()
    if gold != [expected_id, END_ID]:
        raise ValueError('Encoded verdict does not match the independent native label')
    pair = logits[0, slots[0], [FAILURE_ID, SUCCESS_ID]].float()
    pred = bool(pair.argmax().item())
    binary_nll = -torch.log_softmax(pair, dim=0)[int(expected)].item()
    return pred, binary_nll
