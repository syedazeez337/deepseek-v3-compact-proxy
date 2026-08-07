# DeepSeek-V3 Compact Proxy

A from-scratch implementation of DeepSeek-V3's three transferable architecture mechanisms, sized to train on one
consumer GPU:

- **Multi-head Latent Attention (MLA)** with decoupled RoPE and weight-absorbed decode
- **DeepSeekMoE**: shared plus fine-grained routed experts, auxiliary-loss-free load balancing
- **Multi-Token Prediction (MTP)**: a depth-1 draft head trained alongside the main objective

This does not load or reproduce the official 671B weights. The goal is architectural fidelity at a scale that
fits a 6GB laptop GPU.

Every component went through the same loop, recorded one file per gate in `experiments/`:

```text
specification -> implementation -> tests -> controlled experiment -> measured result -> interpretation
```

Hyperparameters published by DeepSeek are treated as a starting prior rather than a given. Several gates below
measured a value tuned at V3's scale performing worse here, and adopted the smaller value that won. Gates that
failed, or that found bugs invalidating an earlier conclusion, are recorded alongside the ones that worked.

## Current status

Phase 1, pretraining architecture, is complete through Gate W.

**Best result: 41.35 validation perplexity on WikiText-103**, from `compact_v3_wikitext103_2ep.pt` (229.4M
tokens, 2 epochs, 64,000 steps). Evaluated deterministically over all 470 validation windows.

Configuration B:

```text
d_model 512, n_heads 8, n_layer 6 (1 dense prefix + 5 MoE), context 520
n_routed_experts 32, n_shared_experts 1, top_k 2 (about 6.25% density)
q_lora_rank 128, kv_lora_rank 128, qk_nope_head_dim 48, qk_rope_head_dim 16, v_head_dim 48
route_scale 0.75, router_bias_update_rate 1e-4
mtp_weight 0.3, annealed to 0.1 at 67.6% of training
155,271,168 parameters total, about 36M active per token
```

Two caveats on that perplexity number. It is not comparable to the earlier 292.54 figure, which used WikiText-2
validation with a WikiText-2 tokenizer at context 264, a different corpus and denominator. The comparable
predecessor is Gate V's 87.85 on the identical setup, which makes Gate W 2.12x better. And at a measured
fertility of 1.13 tokens per word, 41.35 subword perplexity is 67.1 word-level, against 29.4 for a published
6-layer 156M-parameter decoder. This model is roughly 2.3x off a well-trained model of the same shape.

The model writes locally coherent Wikipedia-style prose and picks up specifics from the prompt. It drifts across
paragraphs and invents facts. It has had no instruction tuning, so it continues text and does not answer
questions.

## Install

`uv` handles Python and every dependency, so this is the only thing to install first.

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell, then clone and set up:

```bash
git clone https://github.com/AI-Yatra/deepseek-v3-compact-proxy.git
cd deepseek-v3-compact-proxy
uv sync
```

`uv sync` fetches Python 3.13 and everything else, including the right PyTorch build for your platform: CUDA 13.0
on Windows and Linux, the PyPI build on macOS, which runs on CPU or MPS. Nothing else to configure.

Check it works:

```bash
uv run pytest -q tests
```

90 tests, around a minute. CUDA is optional and GPU tests skip without it.

## Get a checkpoint

Checkpoints live on Hugging Face:
**[huggingface.co/syedazeez/deepseek-v3-compact-proxy](https://huggingface.co/syedazeez/deepseek-v3-compact-proxy)**

`compact_v3_wikitext103_2ep.pt` is the one to want: 41.35 perplexity, the Gate W result. Download it together
with `tokenizer_wikitext103.json`, which it was trained with. Decoding with a different tokenizer produces text
that looks like noise rather than raising an error.

Same command on every platform:

```bash
uv run pip install "huggingface_hub[cli]"
uv run huggingface-cli download syedazeez/deepseek-v3-compact-proxy compact_v3_wikitext103_2ep.pt tokenizer_wikitext103.json --local-dir checkpoints
```

Or without the CLI:

```bash
uv run python -c "from huggingface_hub import hf_hub_download as d; [d('syedazeez/deepseek-v3-compact-proxy', f, local_dir='checkpoints') for f in ['compact_v3_wikitext103_2ep.pt','tokenizer_wikitext103.json']]"
```

Both files land in `checkpoints/`. The download is 1.8GB, because checkpoints carry AdamW optimizer state for
resuming training; the weights alone are 592MB. `checkpoints/README.md` lists every checkpoint, the gate that
produced it, and the command that regenerates it.

## Serve it in a browser

```bash
uv run python serve.py --checkpoint checkpoints/compact_v3_wikitext103_2ep.pt
```

Open <http://127.0.0.1:8000>. Ctrl-C stops it. Forward slashes work on every platform, including PowerShell.

The tokenizer is found automatically: the path recorded in the checkpoint, then
`checkpoints/tokenizer_wikitext103.json`, then the corpus caches. Pass `--tokenizer` only to override that. A
wrong path fails loudly and lists every tokenizer it did find.

| flag | default | notes |
|---|---|---|
| `--checkpoint` | `checkpoints/compact_v3_wikitext103_shakedown.pt` | any `.pt` file |
| `--device` | `cuda` when available | `cpu` works everywhere, including Macs |
| `--port` | `8000` | |
| `--tokenizer` | resolved from the checkpoint | override the search |

Running on CPU costs little. Single-token decode is latency-bound rather than compute-bound, so CPU measures
about 65 tok/s against 49 tok/s on GPU. Use it when the GPU is busy training:

```bash
uv run python serve.py --checkpoint checkpoints/compact_v3_wikitext103_2ep.pt --device cpu
```

The page is a completion playground rather than a chat window, because a base model with no instruction tuning
continues text instead of answering questions. It streams tokens as they generate and gives you:

- **settings**: temperature, top-k, top-p, max tokens, persisted for the session
- **sampling / greedy**: switch decoding mode
- **raw**: monospace with whitespace made visible, for inspecting tokenization
- **tokens**: a collapsible panel showing each generated token, with its id on hover
- a context meter reading `7 prompt + 96 new / 520 ctx`, which flags when a request was capped
- a stop button that keeps whatever generated so far

Temperature between 0.7 and 0.9 works well. Greedy decoding tends to loop at this scale.

## Complete a prompt from the command line

**Windows (PowerShell):**

```powershell
uv run python complete.py "The bridge was built in" `
  --checkpoint checkpoints/compact_v3_wikitext103_2ep.pt `
  --max-new-tokens 40 --do-sample --temperature 0.8 --top-k 40
```

**macOS and Linux:**

```bash
uv run python complete.py "The bridge was built in" \
  --checkpoint checkpoints/compact_v3_wikitext103_2ep.pt \
  --max-new-tokens 40 --do-sample --temperature 0.8 --top-k 40
```

Prints JSON with the prompt, the completion, the continuation alone, and tokens per second. Add `--device cpu`
on a machine without CUDA.

## Train

**Windows (PowerShell):**

```powershell
uv run python v3_cli.py `
  --real-corpus --dataset-config wikitext-103-raw-v1 --dataset-cache-dir data_v3_103 `
  --steps 64000 --warmup-steps 1000 `
  --batch-size 7 --sequence-length 512 `
  --checkpoint-every 4000 --eval-batches 68 --generate 64 `
  --device cuda --enable-mtp `
  --checkpoint checkpoints/my_run.pt
```

**macOS and Linux:**

```bash
uv run python v3_cli.py \
  --real-corpus --dataset-config wikitext-103-raw-v1 --dataset-cache-dir data_v3_103 \
  --steps 64000 --warmup-steps 1000 \
  --batch-size 7 --sequence-length 512 \
  --checkpoint-every 4000 --eval-batches 68 --generate 64 \
  --device cuda --enable-mtp \
  --checkpoint checkpoints/my_run.pt
```

First run downloads WikiText-103, fits a 32K byte-level BPE tokenizer on the train split alone, and caches both
with SHA256 provenance in `data_v3_103/metadata.json`. That takes several minutes.

Add `--resume` to continue an interrupted run from its checkpoint. Optimizer state, GradScaler, RNG state and
data-provider position all restore, so a resumed run continues rather than restarting. Checkpoint writes go to a
temp file and rename into place, so an interrupted write cannot destroy the previous checkpoint.

`v3_cli.py --help` lists every flag. Each was added by a specific gate and defaults to that gate's measured-best
value.

Gate W took about 14 hours on an RTX 3050 6GB at batch 7 and sequence 512, across an interruption and resume,
with throughput degrading part way through for reasons recorded in its gate document. Batch 8 exceeds VRAM on
that card and runs slower, because Windows spills to shared system memory rather than raising an
out-of-memory error. Training needs CUDA; a Mac can serve a checkpoint but not train one at this scale.

## Module map

The library is `src/compact_v3/`. The three CLI entry points stay at the repo root.

| File | Contents |
|---|---|
| `src/compact_v3/config.py` | `CompactV3Config`, every architectural knob, plus legacy-aware checkpoint loading |
| `src/compact_v3/mla.py` | Multi-head Latent Attention: fused `reference`, `prefill`, absorbed `decode`, and two manual variants kept as equivalence proofs |
| `src/compact_v3/rope.py` | Decoupled RoPE |
| `src/compact_v3/norms.py` | RMSNorm |
| `src/compact_v3/routing.py` | Sigmoid-affinity top-k router, loss-free bias balancer, load-entropy diagnostics |
| `src/compact_v3/experts.py` | SwiGLU expert |
| `src/compact_v3/moe.py` | DeepSeekMoE: shared plus routed experts, sorted dispatch, sequence-balance loss |
| `src/compact_v3/block.py` | Pre-norm MLA plus dense or MoE residual block |
| `src/compact_v3/model.py` | Embedding, blocks, tied output head, MTP wiring |
| `src/compact_v3/mtp.py` | Depth-1 MTP objective and weight-annealing schedule |
| `src/compact_v3/generation.py` | Cached and uncached generation, proven token-identical |
| `src/compact_v3/training.py` | AdamW with warmup and cosine decay, FP16 autocast, atomic checkpointing |
| `src/compact_v3/data.py` | WikiText loader, train-split tokenizer, deterministic evaluation |
| `v3_cli.py` | Train, checkpoint, resume, generate |
| `complete.py` | Prompt in, text out |
| `serve.py` | Playground server, streams over SSE, standard library only |
| `ui/index.html` | Browser playground |
| `experiments/` | One document per gate: specification, method, numbers, interpretation |
| `tests/` | 90 tests |

Gate documents name modules by the flat filenames they had before the package restructure (`v3_config.py`,
`moe_v3.py`, and so on). Those names are left as written because they record what was true at the time. The
table above maps them to current paths.

## Gate history

| Gate | Result |
|---|---|
| A | MLA reference and cache equivalence |
| B | Shared plus routed MoE mechanics |
| C | Decoder integration |
| D | Depth-1 MTP |
| E | Cached and uncached generation proven token-identical |
| F | AMP training and checkpoint mechanics |
| G | Full train, checkpoint, resume, generate lifecycle |
| H | WikiText-2 corpus smoke test, provenance hashing |
| I | 1M-token dense control, PPL 598.27 |
| J | 1M-token dense against 4-expert top-1 MoE |
| K | Load-entropy diagnostic; found a layer collapsing to 0.32 |
| L | Swept bias update rate; 1e-4 beat the literature's 1e-3 at this scale |
| M | Top-2 routing, 2.7% better perplexity for zero extra stored parameters |
| N | Dense-layer prefix compute-matched to MoE active FLOPs; first attempt was confounded |
| O | Scale-up to Configuration B, 15M to 155M parameters |
| P | Swept `route_scale`; V3's own 2.5 measured worst, 0.75 won |
| Q | MTP weight annealing; conclusion later invalidated by Gate U |
| R | MLA weight absorption matched to V3's inference code; fixed a VRAM bug that had produced a wrong conclusion in Gate O |
| S | Fixed a tokenizer decoder bug that had blocked all readable output since Gate H |
| T | Sorted MoE dispatch, 25.7% faster training and 1.9x faster decode, bit-exact. The textbook grouped-GEMM fix measured worse and was rejected |
| U | Pre-run audit: found the MTP objective was degenerate, checkpoint writes were not atomic, validation noise exceeded the effects two gates had reported, and 8 of 15 checkpoints had been unloadable since Gate N |
| V | WikiText-103 shakedown, PPL 270.99 to 87.85; fused attention, 3.10x on the MLA layer |
| W | **2 epochs of WikiText-103, 229.4M tokens, validation perplexity 41.35** |

Measured numbers and citations are in each `experiments/GATE_<letter>_*.md`.

## Roadmap

**Serving.** The 1.8GB checkpoint is 1185MB of optimizer state. An inference-only export is 592MB, and int8 with
a scale per 32 values is about 165MB. Round-tripping this model's weights through int8 gives 0.46% to 0.93%
relative error by tensor group, so the perplexity cost needs measuring against the 41.35 baseline before
anything ships.

**MTP speculative decoding.** Gate U fixed the objective and Gate W confirmed it works: the MTP-to-main loss
ratio rose from 1.033 to 1.153 across the run, which is what a working second-token predictor does. DeepSeek
reports 85-90% draft acceptance and 1.8x tokens per second from this mechanism.

**Two defects Gate W exposed.** `train_tokens.pt` stores 115M tokens as int64, using 922MB where uint16 would
use 230MB; the paging that caused halved throughput mid-run. And `v3_cli.py` writes its final checkpoint from
training history, so it carries no validation perplexity.

**Phase 2, YaRN context extension.** Staged fine-tuning applied to the decoupled RoPE key alone. V3's `s=40,
a=1, b=32` need scaling down from its 4K to 128K schedule.

**Phase 3, SFT and GRPO post-training.** Needs an instruction-tuning dataset chosen with the same provenance
diligence as WikiText-2, and a verifiable-reward task sized to what a 155M-parameter model can do.

## Environment

Windows 11, Python 3.13, PyTorch 2.13.0+cu130, RTX 3050 Laptop 6GB (Ampere, compute capability 8.6). `uv`
manages the environment through `pyproject.toml` and `uv.lock`.

## License

Apache 2.0, see `LICENSE`. Checkpoints on Hugging Face carry the same license. WikiText is licensed separately
as CC BY-SA 4.0 / GFDL, listed on the
[Salesforce/wikitext dataset card](https://huggingface.co/datasets/Salesforce/wikitext).
