---
license: apache-2.0
language:
- en
library_name: pytorch
pipeline_tag: text-generation
datasets:
- Salesforce/wikitext
tags:
- deepseek
- deepseek-v3
- mixture-of-experts
- multi-head-latent-attention
- multi-token-prediction
- research
---

# DeepSeek-V3 Compact Proxy — Checkpoints

Trained checkpoints for [deepseek-v3-compact-proxy](https://github.com/syedazeez337/deepseek-v3-compact-proxy),
a mechanism-faithful proxy for DeepSeek-V3's architecture (Multi-head Latent Attention, DeepSeekMoE, Multi-Token
Prediction) — not a reproduction of the 671B model. Every checkpoint here corresponds to a specific, documented
experiment ("gate") in that repository; each gate's specification, implementation, tests, measured results, and
interpretation live in its `experiments/GATE_*.md` file.

Checkpoints are hosted here rather than in the code repository because GitHub blocks any file over 100MB, and
several of these (Configuration B, with optimizer state) are close to 2GB each.

Every checkpoint below is also reproducible from the exact command recorded in its gate document. This Hugging
Face copy is a convenience rather than the only record.

## Which one to use

**`compact_v3_wikitext103_2ep.pt`** is the best model here: 41.35 validation perplexity on WikiText-103, from
229,376,000 tokens (2 epochs, 64,000 steps). Download it together with `tokenizer_wikitext103.json`, which it
was trained with. Decoding with a different tokenizer produces noise.

```powershell
uv run huggingface-cli download syedazeez/deepseek-v3-compact-proxy `
  compact_v3_wikitext103_2ep.pt tokenizer_wikitext103.json --local-dir checkpoints

uv run python serve.py --checkpoint checkpoints\compact_v3_wikitext103_2ep.pt `
  --tokenizer checkpoints\tokenizer_wikitext103.json
```

That opens a browser playground at http://127.0.0.1:8000. This is a base model with no instruction tuning, so it
continues text rather than answering questions.

The 15 Configuration A and B checkpoints below were trained on WikiText-2 with a different tokenizer, which is
not uploaded here. Regenerate it by running `v3_cli.py --real-corpus` once, which fits it from the train split.

## Configuration B, WikiText-103 (155M parameters, Gates V-W)

| File | Gate | Result | Doc |
|---|---|---|---|
| `compact_v3_wikitext103_2ep.pt` | W | 2 epochs, 229.4M tokens, **41.35 PPL** | `experiments/GATE_W_WIKITEXT103_2EPOCH.md` |

Trained at batch 7, sequence 512, context 520, with MTP enabled and the loss weight annealed 0.3 to 0.1. The
earlier 292.54 figure quoted for Gate Q is not comparable: it used WikiText-2 validation with a WikiText-2
tokenizer at context 264.

## Configuration A (~15M parameters, Gates I-N)

| File | Gate | Config | Doc |
|---|---|---|---|
| `compact_v3_wikitext_dense_1m.pt` | I | Dense control, MoE disabled | `experiments/GATE_I_DENSE_CONTROL_1M.md` |
| `compact_v3_wikitext_moe_1m.pt` | J | 4-expert top-1 MoE | `experiments/GATE_J_MOE_COMPARISON_1M.md` |
| `compact_v3_wikitext_moe_1m_gatek.pt` | K | Same as J, entropy-logging rerun | `experiments/GATE_K_LOAD_BALANCING.md` |
| `compact_v3_wikitext_moe_1m_gatel_u00001.pt` | L | `router_bias_update_rate=0.0001` (new default) | `experiments/GATE_L_BIAS_RATE_SWEEP.md` |
| `compact_v3_wikitext_moe_1m_gatel_u003.pt` | L | `router_bias_update_rate=0.003` | `experiments/GATE_L_BIAS_RATE_SWEEP.md` |
| `compact_v3_wikitext_moe_1m_gatel_u01.pt` | L | `router_bias_update_rate=0.01` | `experiments/GATE_L_BIAS_RATE_SWEEP.md` |
| `compact_v3_wikitext_moe_1m_gatem_topk2.pt` | M | `top_k=2` (new default) | `experiments/GATE_M_TOP2_ROUTING.md` |
| `compact_v3_wikitext_moe_1m_gaten_dense1.pt` | N | `n_dense_layers=1`, confounded first attempt (kept for the record) | `experiments/GATE_N_DENSE_LAYER_PREFIX.md` |
| `compact_v3_wikitext_moe_1m_gaten_dense1_matched.pt` | N | `n_dense_layers=1`, corrected (new default) | `experiments/GATE_N_DENSE_LAYER_PREFIX.md` |

## Configuration B (~155M parameters, Gates O-Q)

| File | Gate | Config | Doc |
|---|---|---|---|
| `compact_v3_wikitext_moe_scaleup_1m.pt` | O | Scale-up baseline, `route_scale=1.0` | `experiments/GATE_O_GPU_SCALEUP.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gatep_rs075.pt` | P | `route_scale=0.75` (new default) | `experiments/GATE_P_ROUTE_SCALE.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gatep_rs15.pt` | P | `route_scale=1.5` | `experiments/GATE_P_ROUTE_SCALE.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gatep_rs25.pt` | P | `route_scale=2.5` (V3's own value) | `experiments/GATE_P_ROUTE_SCALE.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_const.pt` | Q | MTP enabled, constant weight 0.3 | `experiments/GATE_Q_MTP_ANNEALING.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_anneal.pt` | Q | MTP enabled, annealed 0.3->0.1 (292.54 PPL on WikiText-2) | `experiments/GATE_Q_MTP_ANNEALING.md` |

Gate R (MLA weight absorption) is an inference-path code change validated by unit tests and benchmarks; it did
not produce a training checkpoint.

## Regenerating a checkpoint

Each gate document's "Experiment" section has the exact command. For example, Gate Q's best checkpoint:

```powershell
uv run python v3_cli.py --real-corpus --steps 1954 --checkpoint-every 250 --eval-batches 32 --generate 32 --device cuda --enable-mtp --checkpoint checkpoints/compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_anneal.pt
```
