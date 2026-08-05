# Gate O — GPU utilization scale-up (Configuration B)

## Status

Completed on 2026-08-04. New default configuration adopted, superseding Configuration A for all future gates.

## Motivation

Every gate through N ran at Configuration A (d_model=256, 4 layers, 4 experts, ~15M parameters), which measured
at only **9.5% of the RTX 3050's 6GB VRAM** during real training (batch 8, sequence 64). Before continuing the
DeepSeek-V3 fidelity roadmap, the question was how much of the GPU's actual capacity was being left unused, and
which scaling lever uses it best.

## Method

Empirically probed VRAM/throughput rather than estimating from parameter-count formulas alone, since a first pass
gave a misleading signal: Windows silently fell back to slow shared system memory for a 9.5GB-allocated config,
making it "succeed" at 8x worse throughput than a 6.2GB run instead of raising an out-of-memory error. Capping
`torch.cuda.set_per_process_memory_fraction(0.90)` exposed the true ceiling and was used for all subsequent probes.

Findings across three scaling levers (see raw sweep data in session record, `probe_vram*.py` in scratch):

| Lever | Effect |
|---|---|
| Batch size / context length (same tiny model) | 16x throughput (1055 -> 16,624 tok/s) for 3x VRAM — cheap, was leaving most GPU parallelism idle |
| `d_model`/`n_layer`/`n_heads` | Stored and active compute scale together — most expensive capacity per VRAM byte |
| `n_routed_experts` (top_k fixed) | Cheapest VRAM cost per parameter (matches V3's sparse-MoE philosophy), but exposed a real bottleneck: `moe_v3.py` dispatches experts in a Python loop, so throughput drops as expert count grows even though active compute per token doesn't change |

Given a choice between throughput-first, balanced, or fidelity-first scaling, the user chose to prioritize expert
count — moving closer to DeepSeekMoE's actual fine-grained-sparse character — accepting the dispatch-loop
throughput cost as a known limitation for a later efficiency gate (adjacent to the MLA weight-absorption gate
already queued).

## Configuration B (new default)

```text
d_model: 256 -> 512        n_layer: 4 -> 6         n_heads: 4 -> 8
q_lora_rank: 64 -> 128     kv_lora_rank: 64 -> 128
qk_nope_head_dim: 32 -> 48 qk_rope_head_dim: 16 (unchanged)  v_head_dim: 32 -> 48
n_routed_experts: 4 -> 32  n_shared_experts: 1 (unchanged)  top_k: 2 (unchanged, now ~6.25% density)
expert_hidden_dim: 384 -> 512   n_dense_layers: 1 (unchanged)   context_length: 256 -> 264
CLI defaults: --batch-size 1->8   --sequence-length 32->256
```

Measured at this exact configuration (synthetic smoke, batch 8, sequence 256): 155,271,168 unique parameters,
4,058.4 MB peak training VRAM (66.1% of 6GB, comfortable headroom under the 90% cap), ~4,088 tokens/sec.

## Implementation

- `v3_config.py`: all `CompactV3Config` defaults updated to Configuration B above.
- `v3_cli.py`: `--batch-size` and `--sequence-length` argparse defaults updated to match.
- No test constructed `CompactV3Config()` with zero overrides for assertions tied to the old scale (checked via
  grep before changing), so the full active suite needed no changes: 54 passed.
- Verified CUDA prefill/decode equivalence holds at the new scale (155M params) before running real data.

## Real-corpus validation run

Same WikiText-2 dataset/tokenizer/provenance as every prior real-corpus gate. At the new batch(8)/context(256),
1,954 steps now covers ~4.0M tokens instead of the previous ~1.0M (2048 tokens/step vs 512) — intentional, more
tokens for a ~10x larger model, not a deviation from the step-count convention.

```text
Checkpoint: checkpoints/compact_v3_wikitext_moe_scaleup_1m.pt (1.75 GB on disk)
Steps: 1,954   Tokens: 4,001,792   Precision: FP16 autocast + GradScaler
```

## Results

| Metric | Configuration A best (Gate N, 14.8M params) | Configuration B (Gate O, 155.3M params) |
|---|---:|---:|
| Final validation perplexity | 581.57 | **298.88** |
| Final validation loss | 6.366 | 5.700 |
| Unique parameters | 14,821,120 | 155,271,168 |
| Generation tokens/sec | 76.38 | 29.94 |
| Generation peak VRAM | 566.35 MB (9.2%) | **5,869.0 MB (95.5%)** |
| Training peak VRAM (measured separately) | ~583 MB (9.5%) | ~4,058 MB (66.1%) |

Perplexity roughly halved (298.88 vs 581.57) from more parameters and more training tokens together — expected
and not a controlled single-variable result (both scale and token budget changed at once by design). Per-layer
load entropy stayed reasonable across all 5 MoE layers through the whole run (never collapsed toward 0; lowest
observed was 0.679 early in training, settling to 0.81-0.99 by the final checkpoint) even though
`router_bias_update_rate=1e-4` was tuned at 4 experts in Gate L, not retuned for 32 — a positive but unconfirmed
generalization, not yet stress-tested.

## Finding: generation VRAM is now tight

Training peaked at 66% VRAM, but cached generation peaked at **95.5%** — close to the card's real limit. This is
expected given `mla.py`'s `decode()` still reconstructs full K/V from the compressed cache at every step (weight
absorption not yet implemented, tracked as a queued gate) — at 155M parameters and 32 experts per layer this
reconstruction cost is no longer negligible. This raises the priority of MLA weight absorption relative to where
it sat before, since it now also protects against a real generation-time OOM risk, not just decode speed.

## Interpretation

The GPU had an order of magnitude of unused headroom through Gate N. Scaling to Configuration B used it
deliberately along the axis most faithful to DeepSeek-V3's actual design (many sparse experts, modest top-k
density), at real measured cost (dispatch-loop throughput, tighter generation VRAM) that were accepted knowingly
rather than discovered by accident. Configuration B is now the baseline for the rest of the fidelity roadmap.

## Next gate

Continue the "true to DeepSeek-V3" roadmap at Configuration B. Route-scale retuning (previously planned as
"Gate O") is renumbered to **Gate P**, MTP loss-weight annealing to **Gate Q**, MLA weight absorption to
**Gate R** — the last of these gained urgency from the generation-VRAM finding above. A future efficiency gate
(batched/grouped expert dispatch, replacing the current per-expert Python loop) is now explicitly noted as
worth doing before pushing expert count further.
