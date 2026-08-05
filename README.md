# DeepSeek-V3 Compact Proxy

A clean-start, mechanism-faithful proxy for DeepSeek-V3's architecture: Multi-head Latent Attention (MLA),
DeepSeekMoE (shared + fine-grained routed experts, auxiliary-loss-free load balancing), and Multi-Token
Prediction (MTP). It does not load or reproduce the official 671B model — the goal is architectural fidelity at a
scale that trains on a single consumer GPU, not literal parameter-count reproduction.

Every component follows a research-engineering discipline, recorded in `experiments/GATE_*.md`:

```text
mathematical specification -> reference implementation -> tests -> controlled experiment
  -> measured result -> interpretation -> next change
```

Hyperparameters copied from DeepSeek-V3's own published values are treated as a *prior*, not a given — several
gates below found that a value tuned at V3's scale (256 experts, huge batches) actively hurt at this project's
scale, and the smaller value that measured better became the new default. Failures are recorded, not hidden:
several gates below describe an approach that didn't work, or a bug that was found and fixed, alongside what did.

## Current status (2026-08-04)

Phase 1 of the fidelity roadmap — pretraining architecture — is complete through Gate R. The model is at
**Configuration B**:

```text
d_model=512  n_heads=8  n_layer=6 (1 dense prefix + 5 MoE)
n_routed_experts=32  n_shared_experts=1  top_k=2 (~6.25% density)
q_lora_rank=128  kv_lora_rank=128  qk_nope_head_dim=48  qk_rope_head_dim=16  v_head_dim=48
route_scale=0.75  router_bias_update_rate=0.0001
mtp_weight=0.3 annealed to 0.1 at 67.6% of training (V3's own schedule)
~155M parameters, absorbed MLA decode (matches V3's real inference/model.py "absorb" path)
```

Best measured result: **292.54 validation perplexity** on WikiText-2 (`checkpoints/..._gateq_mtp_anneal.pt`,
hosted on Hugging Face — see below). At this perplexity the model produces grammatically fluent but topically
incoherent completions — see `experiments/GATE_S_READABLE_GENERATION.md` for a real example and honest
interpretation of what "ready" would actually require.

**What's next** (not yet started, three options on the table, see "Roadmap" below): Phase 2 (YaRN context
extension), Phase 3 (SFT + GRPO-based RL post-training), or a flagged efficiency fix to `src/compact_v3/moe.py`'s per-expert
dispatch loop.

## Quickstart

```powershell
uv sync
uv run pytest -q tests                          # full active suite (tests/, not archive/)
uv run python v3_cli.py --real-corpus --steps 1954 --checkpoint-every 250 --eval-batches 32 --generate 32 --device cuda --enable-mtp --checkpoint checkpoints/my_run.pt

# complete a prompt with a checkpoint (download one from the Hugging Face repo below first)
uv run python complete.py "The cat sat on" --checkpoint checkpoints/<name>.pt --device cuda
```

`v3_cli.py --help` lists every flag; each one was added by a specific gate below and defaults to that gate's
measured-best value. `--real-corpus` downloads and caches WikiText-2 on first use (provenance hashes recorded in
`data_v3/metadata.json`). Requires CUDA for a GPU run; `--device cpu` works for correctness checks at small scale.

## Module map

The library lives in `src/compact_v3/`; the two CLI entry points stay at the repo root.

| File | What it is |
|---|---|
| `src/compact_v3/config.py` | `CompactV3Config` — every architectural knob, current defaults = Configuration B |
| `src/compact_v3/mla.py` | Multi-head Latent Attention: `reference`, `prefill`, absorbed `decode` (+ `decode_naive` for the equivalence proof) |
| `src/compact_v3/rope.py` | Decoupled RoPE |
| `src/compact_v3/norms.py` | RMSNorm |
| `src/compact_v3/routing.py` | Sigmoid-affinity top-k router, auxiliary-loss-free bias load balancer, load-entropy diagnostics |
| `src/compact_v3/experts.py` | SwiGLU expert |
| `src/compact_v3/moe.py` | DeepSeekMoE: shared + routed experts, sequence-balance loss |
| `src/compact_v3/block.py` | Pre-norm MLA + (dense or MoE) residual block, per-layer dense/MoE override |
| `src/compact_v3/model.py` | Full model: embedding -> blocks -> tied output head |
| `src/compact_v3/mtp.py` | Sequential depth-1 MTP objective, weight-annealing schedule |
| `src/compact_v3/generation.py` | Cached vs uncached generation, proven token-identical |
| `src/compact_v3/training.py` | AdamW + warmup/cosine LR, FP16 autocast + GradScaler, full checkpoint/resume |
| `src/compact_v3/data.py` | WikiText loader: train-only BPE tokenizer, SHA256-hashed provenance |
| `v3_cli.py` | End-to-end train -> checkpoint -> resume -> generate CLI |
| `complete.py` | Prompt-in/text-out CLI: encode a real prompt, generate, decode to readable text |
| `experiments/` | One `GATE_<letter>_*.md` per gate: spec, method, measured numbers, interpretation |
| `tests/` | Active test suite (66 tests as of Gate S) |
| `archive/` | A prior, superseded project attempt — excluded from this repo; see git history if needed |

Gate documents in `experiments/` refer to modules by the flat, pre-package names they had when each gate was
run (`v3_config.py`, `moe_v3.py`, `compact_v3_model.py`, and so on). Those references are left as written: they
are a record of what was true at the time, not live links. The table above maps each one to its current path.

## Gate history

| Gate | What it settled |
|---|---|
| A | Compact MLA reference/cache equivalence |
| B | Shared-plus-routed MoE mechanics |
| C | Decoder integration (full model) |
| D | Sequential depth-1 MTP |
| E | Cache-aware generation, cached vs uncached token-identical |
| F | AMP training/checkpoint mechanics |
| G | Full train -> checkpoint -> resume -> generate lifecycle |
| H | WikiText-2 real-corpus smoke, provenance hashing |
| I | 1M-token dense control |
| J | 1M-token dense vs 4-expert top-1 MoE |
| K | Load-balancing entropy diagnostic added; found persistent imbalance |
| L | Bias-update-rate swept against DeepSeek's Loss-Free-Balancing paper; lower rate won, opposite of the literature's own scale |
| M | Top-2 routing; +2.7% perplexity for zero extra stored parameters |
| N | Dense-layer prefix, compute-matched to MoE active FLOPs (first attempt was confounded — see the doc) |
| O | GPU scale-up: Configuration A -> B (15M -> 155M params); found & used only 9.5% of the GPU before this |
| P | `route_scale` swept; V3's own value (2.5) measured worse than a smaller one (0.75) |
| Q | MTP loss-weight annealing (V3's 0.3->0.1 schedule); confirmed beneficial |
| R | MLA weight absorption, matched to V3's real inference code; also found and fixed a `v3_cli.py` VRAM bug that had produced a wrong conclusion in Gate O |
| S | Fixed a tokenizer decoder bug that had blocked all readable output; added `complete.py`; first real prompt-in/text-out check on the best checkpoint |

Full detail, measured numbers, and citations are in each `experiments/GATE_<letter>_*.md`.

## Roadmap

Three directions are open, not yet started:

1. **Phase 2 — YaRN context extension.** Two staged fine-tuning phases extending context, applied only to the
   decoupled RoPE key (matches this project's MLA split). V3's `s=40, α=1, β=32` need scaling down from its
   4K->32K->128K schedule to this project's much smaller base context.
2. **Phase 3 — SFT + GRPO-based RL post-training.** Needs two decisions first: an instruction-tuning dataset for
   SFT (with the same provenance diligence as WikiText-2), and a verifiable-reward toy task for RL sized to what
   a 155M-parameter model can actually do (real math/code reasoning is out of reach even at this scale) —
   R1-Zero's rule-based accuracy+format reward, not a learned reward model.
3. **MoE dispatch efficiency.** `src/compact_v3/moe.py` dispatches routed experts in a Python loop; Gate O found this costs
   real throughput as expert count grows even though active compute/token doesn't change. Should happen before
   expert count is pushed past 32.

## Checkpoints

Trained checkpoints (up to ~1.9GB each with optimizer state) are hosted on Hugging Face Hub, not in this repo —
see `checkpoints/README.md` for the full list and the exact command to regenerate any of them.

## Environment

Developed on Windows 11, Python 3.13, PyTorch 2.13.0+cu130, RTX 3050 Laptop 6GB (Ampere, compute capability 8.6).
`uv` manages the environment (`pyproject.toml` + `uv.lock`).

## License

Code is licensed under Apache 2.0 (see `LICENSE`). Trained checkpoints on Hugging Face Hub carry the same
license. The WikiText-2 training data itself is separately licensed CC BY-SA 4.0 / GFDL as listed by the
[Salesforce/wikitext dataset card](https://huggingface.co/datasets/Salesforce/wikitext) — unaffected by the
code/weights license above.
