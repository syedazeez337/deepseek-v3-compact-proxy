# Gate R — MLA weight absorption, and a correction to Gate O

## Status

Completed on 2026-08-04. `decode()` now uses the absorbed computation; the un-absorbed path is kept as
`decode_naive()` for the equivalence proof. A real memory-management bug in `v3_cli.py`, found while
investigating this gate, is also fixed.

## Research basis

Fetched DeepSeek-V3's official `inference/model.py` MLA "absorb" branch directly rather than deriving from the
technical report alone. Confirmed the exact technique: the combined KV up-projection weight is split into a
content part and a value part; the query's non-RoPE content (`q_nope`, already computed per new token) is
absorbed against the content part via `einsum("bshd,hdc->bshc", q_nope, wkv_b_content)`, producing a vector in
compressed (`kv_lora_rank`) space; attention scores are then a direct dot product against the *cached compressed*
KV (`kv_cache`, `pe_cache` — precisely the two tensors this project's `MLACache` already stores), never
reconstructing per-head K. The attention-weighted output is likewise computed by summing over the compressed
cache first, and only projected back to `v_head_dim` afterward via the value part of `wkv_b`. This is an exact
reordering of the same bilinear computation (associativity of matrix multiplication), not an approximation.

Source: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

## Implementation

- `mla.py`: renamed the existing per-step cache reconstruction logic to `decode_naive()` (kept for the
  equivalence proof). Added a new `decode()` using the absorbed computation: `k_content_up.weight` and
  `v_up.weight` are reshaped per-head and used directly as the absorption matrices via `einsum`, operating on
  `query_content` and the cached `compressed_kv` — `k_content_up`/`v_up` are never invoked on the cache at all.
  Since every call site (`v3_block.py`, `compact_v3_model.py`, `v3_generation.py`) calls `attention.decode()` by
  name, this change applies project-wide with no other code changes.
- Added `test_absorbed_decode_matches_naive_decode` and `test_absorbed_decode_matches_naive_over_multiple_steps`
  (`tests/test_mla.py`) — exact numerical equivalence, single-step and across 4 chained decode calls. The
  pre-existing `test_cached_decode_matches_full_recomputation` (decode vs `reference()`) and the CUDA/FP16
  cached-generation tests (`tests/test_generation_cache.py`) now exercise the absorbed path automatically, since
  they call `decode()` by name. Full active suite: 63 passed.

## A finding that reopened Gate O's own conclusion

Before benchmarking, `mla.py`'s `decode_naive()` was checked for a second, distinct inefficiency: it calls
`self.k_content_up(compressed_cache)` and `self.v_up(compressed_cache)` on the *entire* cache (old + new) at
every single decode step — redundantly reconstructing content K/V for already-seen positions every time, not
just the new token. This looked like it should cost real, measurable time at longer generations.

It didn't. Benchmarking absorbed vs naive decode directly (bypassing the CLI) at Configuration B, generating up
to 512 tokens, showed no meaningful VRAM or throughput difference between the two — both within run-to-run noise
of each other. At this model's scale, the redundant reconstruction is a small fraction of total per-step compute
(dominated by the MoE/attention/dispatch work already present each layer) and doesn't show up as a bottleneck in
eager-mode PyTorch.

That result didn't match Gate O's own finding (cached generation at 95.5% VRAM, presented as evidence that
absorption mattered for memory, not just speed). Testing `generate_cached()` in a clean, single-model process
with no prior training showed genuine generation VRAM was only **~600MB (9.8%)** — nowhere near 95.5%. Tracing
the discrepancy: `v3_cli.py` builds *three* separate `CompactV3Model` instances in one process (`model` for
training, `validation_model` for the eval-batches check, `reload_model` for the final generate) plus their
optimizers, and never frees any of them before the next is created. `evaluate_provider`/`load_checkpoint`'s
`torch.load(..., map_location=device)` also leaves a full duplicate copy of the checkpoint's model/optimizer
state dicts resident on the GPU. None of this is related to MLA weight absorption — it's leftover state from
earlier phases of the same process, still occupying VRAM when `torch.cuda.max_memory_allocated()` reports its
peak during generation. **Gate O's "generation is VRAM-tight" finding was a mismeasurement, not a real property
of decode.**

## Fix (found and corrected in this gate)

`v3_cli.py`: free `model`/`optimizer`/`scaler` after training completes and before validation; free
`validation_model`/`validation_optimizer`/`validation_scaler` after the validation print and before the
reload+generate phase; free `reload_optimizer`/`reload_scaler` and the raw checkpoint `payload` (only its
`step` field is needed) right after loading, before generation runs. Each free is paired with
`torch.cuda.empty_cache()`.

Verified with a real (short, 5-step) CLI run using `--eval-batches` and `--generate`:

```text
Before this gate's fixes (Gate O's real 1954-step run): generation_peak_allocated_mb = 5869.0  (95.5% of 6GB)
After the CLI memory fixes, same real end-to-end flow:  generation_peak_allocated_mb =  610.8  ( 9.9% of 6GB)
```

610.8 MB matches the clean isolated measurement (~600MB) almost exactly, confirming the fix addresses the actual
cause. Full active suite (63 tests) still passes after the fix.

## Interpretation

Two separate things happened in this gate. First, weight absorption itself: implemented faithfully to V3's real
source, proven exactly equivalent to the naive cache-reconstruction path and to full recomputation, and adopted
as `decode()` project-wide — a genuine fidelity improvement (this project's decode path now matches what
DeepSeek's own inference code does), even though it produced no measurable speed or memory benefit at this
model's compact scale. It's still the right implementation to carry forward, since the FLOP savings should
matter more as `qk_nope_head_dim + v_head_dim` grows relative to `kv_lora_rank`, or at much longer contexts —
neither of which this project currently exercises heavily.

Second, and more consequential in practice: the actual VRAM pressure Gate O measured was a real bug — three
un-freed model/optimizer instances stacking up in one process — not evidence that MLA needed absorption for
memory reasons. Fixing it recovered the 6GB card almost entirely (95.5% -> 9.9% for the same real generation
call). This is exactly the kind of thing the project's own discipline exists to catch: measure again before
trusting an earlier conclusion, and record the correction rather than quietly letting it stand.

## Validation

```text
MLA absorption tests: 2 passed (new), existing MLA/generation-cache suite unaffected in behavior
Active V3 suite:       63 passed
CLI memory fix, real 5-step end-to-end run: generation VRAM 610.8MB (was 5869.0MB before the fix)
```

## Next gate

Phase 1 of the pretraining-architecture roadmap (dense-layer prefix, `route_scale`, MTP annealing, MLA
absorption) is now complete. Also still flagged: replacing `moe_v3.py`'s per-expert Python dispatch loop with
batched/grouped dispatch before pushing expert count past 32 (found in Gate O, not yet addressed). Next major
phase per the agreed roadmap is Phase 2 — YaRN-style context extension — or, if a Phase 1 return is preferred
first, the dispatch-loop efficiency work.
