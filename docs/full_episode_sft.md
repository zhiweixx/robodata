# One epoch of Qwen action and outcome SFT

This experiment uses 217 complete, natively scored teacher episodes from
`robodata-lab/robodojo-gpt6-astra-revised-220-20261101` at
`7df93dc28f4e0495812bad917eed6432360f960d`. The other three physical
states have no native label. The committed split binds episode IDs to physical
states. The full manifest contains 205 training episodes and 12 held-out layouts.
The eligible manifest retains 201 complete training episodes (46 successful)
and 11 complete held-out layouts (4 successful). Five scored failures exceed
the pinned model's 262,144-position context:

| Episode | Split | Expanded tokens |
|---|---|---:|
| `imitate_sorting_sequence_s0_l0` | train | 270,194 |
| `imitate_sorting_sequence_s0_l1` | train | 361,556 |
| `imitate_sorting_sequence_s0_l2` | train | 293,154 |
| `imitate_sorting_sequence_s0_l3` | train | 269,075 |
| `make_toast_s0_l2` | held out | 270,923 |

## Prepare on CPU

Use Python 3.12. Download the pinned dataset to an external directory with
your existing Hugging Face access. Keep the original `artifacts/` layout and
release indices. Set `MODEL` to the local BF16 snapshot of
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Keep all data, outputs, experiment logs and checkpoints outside the checkout.

```bash
python -m pip install -e '.[sft-prepare]'
robodata sft prepare --dataset-dir "$DATASET" --model-path "$MODEL" --prepared-dir "$PREPARED"
```

For staging on separate machines, run `prepare --stage reconstruct` with
`--dataset-dir` and `--prepared-dir`. Transfer the verified prepared directory,
then run `prepare --stage encode` with `--prepared-dir` and `--model-path`.
The final `length_report.json` lists expanded Qwen token counts for each full
episode. `manifest.json` keeps all 217 encoded episodes. The selector verifies
the exact five IDs and lengths above against both full-release records and writes
`eligible_manifest.json`. All 212 retained rows reference their original arrays,
image references and split. Any change to the excluded set or its lengths blocks
training. To reuse a previously verified 217-episode encoding, run:

```bash
robodata sft prepare --stage select --prepared-dir "$PREPARED"
```

Check the eligible manifest's fingerprint and counts before taking the GPU
out of service. The training entry point rechecks its derivation.

Every episode contains its ordered observations, camera images and public
teacher replies. A final public-feedback query asks for `success` or
`failure`. Native labels enter only as the answer. The action mask covers valid
actions from successful episodes, with the known execution error removed.
Failed actions and ignored replies stay visible in the causal prefix.
The training split has 603 supervised actions and 201 outcome replies. The
held-out split has 83 supervised actions and 11 outcome replies. Each target
includes its own end-of-message token.

## Train and evaluate

Use an isolated training environment on a single BF16 GPU with the dashboard
and inference services managed separately. Match PyTorch/torchvision and the
CUDA driver for the host. The H200 environment uses PyTorch 2.13.0+cu130,
torchvision 0.28.0+cu130, Transformers 5.19.0, PEFT 0.21.2, Accelerate
1.15.0, W&B 0.31.0 and flash-linear-attention 0.6.0. Install into the training
environment only.

```bash
python -m pip install -e '.[sft]'
robodata sft train --prepared-dir "$PREPARED" --model-path "$MODEL" --run-dir "$RUN"
robodata sft evaluate --prepared-dir "$PREPARED" --model-path "$MODEL" --run-dir "$RUN"
```

After a host interruption, repeat the train command with `--resume` and the
same run directory, manifest and model. The committed optimizer update
determines the next episodes. The final evaluation uses the reloaded adapter.

The language projections use rank-16 LoRA with alpha 32 and dropout 0.
An optimizer batch includes up to eight full episodes. Each episode contributes
its summed action NLL divided by the **batch** action-token count, plus 0.25
times its summed outcome NLL divided by the **batch** outcome-token count.
Each training episode contributes once, across 26 AdamW updates. Gradient
checkpointing and computing vocabulary logits only at the predecessors of
labelled positions keep the full multimodal causal prefix intact.

## Read the metrics

Per-update metrics separate action and outcome NLL, include both token
denominators, cumulative episodes, learning rate and pre-clip gradient norm.
Throughput counts all expanded input positions. The CUDA peak measures
allocator use. Final loss summaries weight by token count. Held-out action
NLL measures imitation. The outcome confusion matrix, balanced accuracy and
binary NLL use the two candidate logits before the gold answer. The report
separates successful layouts, natural failures and give-up failures. Measuring
robot task success requires separate simulator rollouts.

On this split, predicting failure for every visible give-up and success
otherwise gives 8/11 correct and 0.786 balanced accuracy. Use the three
naturally ended failures to check what the adapter learns beyond that signal.

W&B writes offline to the run directory under project `robodojo-qwen-sft`.
After training, use the account authorized for this project:

```bash
wandb login
wandb sync "$OFFLINE_RUN_DIRECTORY"
```

Only send the metrics run. Keep prepared trajectories, images, checkpoints
and adapters in private storage.
