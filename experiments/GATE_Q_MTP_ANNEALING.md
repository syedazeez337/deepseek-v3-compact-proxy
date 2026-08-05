# Gate Q — MTP loss-weight annealing

## Status

Completed on 2026-08-04. Annealed schedule confirmed beneficial and kept as the default `--enable-mtp` behavior.

## Research basis

The DeepSeek-V3 technical report specifies the MTP loss weight `λ` as `0.3` for the first 10T of 14.8T training
tokens (67.6%), then `0.1` for the remaining 4.8T tokens — a single step change, not a smooth interpolation, and
it lands at the same token milestone where V3's own learning-rate schedule transitions from its constant phase
into decay. This project's learning-rate schedule (`learning_rate()` in `v3_training.py`) is a continuous cosine
decay starting right after warmup, with no distinct constant-then-decay phase to reuse as a natural switch point.
The anneal was implemented as the same *fraction of total training steps* (`0.6757 ≈ 10/14.8`) instead, preserving
V3's proportional schedule shape even though the absolute LR curve shape differs.

Source: https://arxiv.org/pdf/2412.19437 (DeepSeek-V3 Technical Report).

## Implementation

- `v3_config.py`: added `mtp_weight_final` (default `0.1`) and `mtp_decay_step_fraction` (default `0.6757`),
  validated (`mtp_weight_final >= 0`, `0 <= mtp_decay_step_fraction <= 1`).
- `mtp.py`: added `mtp_weight_schedule(step, total_steps, initial_weight, final_weight, decay_step_fraction)` — a
  pure step function, matching V3's actual (non-smooth) schedule. `MTPObjective.combined_loss()` now accepts an
  optional `weight` override, defaulting to `config.mtp_weight` for callers that don't care about annealing
  (e.g. `data_v3.py`'s validation-only `evaluate_provider`).
- `v3_training.py`: `train_microbatch` now takes `step`/`total_steps`, computes the annealed weight via the
  schedule, and reports it in its returned metrics (`mtp_weight`) for visibility in checkpoint progress logs.
  `train_steps` threads `step`/`training_config.total_steps` through, consistent with how `learning_rate()`
  already uses `total_steps` for its own schedule.
- `v3_cli.py`: added `--mtp-weight-final` (default `0.1`) and `--mtp-decay-fraction` (default `0.6757`). Setting
  `--mtp-weight-final` equal to `mtp_weight` (0.3) disables annealing for an ablation, without a separate flag.
- Added `test_mtp_weight_schedule_switches_at_decay_fraction`, `test_combined_loss_accepts_weight_override`
  (`tests/test_mtp.py`), `test_mtp_decay_step_fraction_out_of_range_rejected`,
  `test_mtp_weight_final_negative_rejected` (`tests/test_compact_v3_model.py`),
  `test_mtp_annealing_flags_wire_into_config` (`tests/test_v3_cli.py`), and
  `test_mtp_weight_anneals_partway_through_training` (`tests/test_v3_training.py`, confirms the logged
  `mtp_weight` sequence over 5 steps with `decay_step_fraction=0.4` is `[0.3, 0.3, 0.1, 0.1, 0.1]`). Full active
  suite: 61 passed.
- Note: `MTPObjective` is unconditionally constructed in `CompactV3Model.__init__` regardless of `mtp_depth`, so
  its parameters were already counted in the 155M-parameter total before this gate. Enabling MTP only changes
  *activation/backward compute* — measured ~4.6 percentage points more training VRAM at Configuration B
  (49.7% -> 54.3% of 6GB), comfortably within budget.

## Experiment

MTP was disabled by default in every real-corpus gate through P (`--enable-mtp` off, kept ablations matched).
This gate is the first real-corpus run at Configuration B with MTP turned on, so it also serves as the first
real measurement of MTP's own effect, not just the annealing schedule. Two runs, both with `--enable-mtp`,
otherwise identical to Gate P's setup (route_scale=0.75, bias rate 1e-4, top_k=2, n_dense_layers=1, WikiText-2,
batch 8, sequence 256, 1,954 steps, seed 1337):

```text
Constant weight (mtp_weight_final=0.3, i.e. no annealing): checkpoints/compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_const.pt
Annealed weight (defaults: 0.3 -> 0.1 at step ~1321):        checkpoints/compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_anneal.pt
```

## Results

| Condition | Final validation PPL | Final MTP loss | Final `mtp_weight` |
|---|---:|---:|---:|
| No MTP (Gate P baseline) | 296.89 | n/a | n/a |
| MTP, constant weight 0.3 | 294.50 | 0.0185 | 0.3 |
| MTP, annealed 0.3->0.1 | **292.54** | 0.0196 | 0.1 |

## Interpretation

Enabling MTP at all improved validation perplexity over Gate P's no-MTP baseline (294.50 vs 296.89, 0.8% better) —
the auxiliary objective helps the main task here, as V3's own design intends. Annealing the MTP weight down
partway through training improved it further (292.54 vs 294.50, another 0.8% better), confirming V3's schedule
choice is beneficial rather than neutral or harmful at this scale, even though the switch point had to be
translated from "token milestone aligned with an LR phase change" to "step fraction" since this project's LR
schedule has no matching phase structure. The MTP loss itself is small and of similar magnitude in both runs
(0.0185 vs 0.0196), consistent with it being an auxiliary signal — the improvement shows up in main-task
perplexity, not in the MTP loss value.

Annealing was already wired as the default behavior for any future `--enable-mtp` run (`mtp_weight_final=0.1`,
`mtp_decay_step_fraction=0.6757`); this gate's purpose was to verify that default is actually beneficial, not to
select it after the fact.

## Validation

```text
MTP annealing tests:  4 passed
Active V3 suite:      61 passed
Two Configuration B CUDA runs (MTP on): completed, checkpoints preserved
```

## Next gate

Gate R: MLA weight absorption. `mla.py`'s `decode()` still reconstructs full K/V from the compressed cache at
every step; Gate O measured cached generation at 95.5% VRAM at Configuration B, so this is no longer just a
speed optimization. Needs a numerical-equivalence proof against the existing `reference`/`prefill`/`decode` paths
before being trusted.
