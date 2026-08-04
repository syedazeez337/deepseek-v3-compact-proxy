# Checkpoints

Trained checkpoints are hosted on Hugging Face Hub rather than in this repository, since GitHub blocks any file
over 100MB and several of these (Configuration B, with optimizer state) are close to 2GB each.

Hub repo: https://huggingface.co/syedazeez/deepseek-v3-compact-proxy

Every checkpoint below is also fully reproducible from the exact command recorded in its gate document — the
Hugging Face copy is a convenience, not the only record.

## Configuration A (~15M parameters, Gates I-N)

| File | Gate | Config | Doc |
|---|---|---|---|
| `compact_v3_wikitext_dense_1m.pt` | I | Dense control, MoE disabled | `experiments_GATE_I_DENSE_CONTROL_1M.md` |
| `compact_v3_wikitext_moe_1m.pt` | J | 4-expert top-1 MoE | `experiments_GATE_J_MOE_COMPARISON_1M.md` |
| `compact_v3_wikitext_moe_1m_gatek.pt` | K | Same as J, entropy-logging rerun | `experiments_GATE_K_LOAD_BALANCING.md` |
| `compact_v3_wikitext_moe_1m_gatel_u00001.pt` | L | `router_bias_update_rate=0.0001` (new default) | `experiments_GATE_L_BIAS_RATE_SWEEP.md` |
| `compact_v3_wikitext_moe_1m_gatel_u003.pt` | L | `router_bias_update_rate=0.003` | `experiments_GATE_L_BIAS_RATE_SWEEP.md` |
| `compact_v3_wikitext_moe_1m_gatel_u01.pt` | L | `router_bias_update_rate=0.01` | `experiments_GATE_L_BIAS_RATE_SWEEP.md` |
| `compact_v3_wikitext_moe_1m_gatem_topk2.pt` | M | `top_k=2` (new default) | `experiments_GATE_M_TOP2_ROUTING.md` |
| `compact_v3_wikitext_moe_1m_gaten_dense1.pt` | N | `n_dense_layers=1`, confounded first attempt (kept for the record) | `experiments_GATE_N_DENSE_LAYER_PREFIX.md` |
| `compact_v3_wikitext_moe_1m_gaten_dense1_matched.pt` | N | `n_dense_layers=1`, corrected (new default) | `experiments_GATE_N_DENSE_LAYER_PREFIX.md` |

## Configuration B (~155M parameters, Gates O-Q)

| File | Gate | Config | Doc |
|---|---|---|---|
| `compact_v3_wikitext_moe_scaleup_1m.pt` | O | Scale-up baseline, `route_scale=1.0` | `experiments_GATE_O_GPU_SCALEUP.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gatep_rs075.pt` | P | `route_scale=0.75` (new default) | `experiments_GATE_P_ROUTE_SCALE.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gatep_rs15.pt` | P | `route_scale=1.5` | `experiments_GATE_P_ROUTE_SCALE.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gatep_rs25.pt` | P | `route_scale=2.5` (V3's own value) | `experiments_GATE_P_ROUTE_SCALE.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_const.pt` | Q | MTP enabled, constant weight 0.3 | `experiments_GATE_Q_MTP_ANNEALING.md` |
| `compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_anneal.pt` | Q | MTP enabled, annealed 0.3->0.1 (current best, 292.54 PPL) | `experiments_GATE_Q_MTP_ANNEALING.md` |

Gate R (MLA weight absorption) is an inference-path code change validated by unit tests and benchmarks; it did
not produce a training checkpoint.

## Regenerating a checkpoint

Each gate document's "Experiment" section has the exact command. For example, Gate Q's best checkpoint:

```powershell
uv run python v3_cli.py --real-corpus --steps 1954 --checkpoint-every 250 --eval-batches 32 --generate 32 --device cuda --enable-mtp --checkpoint checkpoints/compact_v3_wikitext_moe_scaleup_1m_gateq_mtp_anneal.pt
```
