# Handoff: DeepSeek-V3 by Hand workbook

Paste the block below into a fresh session at the repository root.

---

## Context

I am building an Excel workbook, "DeepSeek-V3 by Hand", modelled on Prof. Tom Yeh's
"Kimi 3 by Hand" seminar sheet (AI by Hand series). Every matrix in the forward pass is
computed live in the grid with dynamic-array formulas, so a reader can trace DeepSeek-V3's
mechanisms cell by cell. It is a companion to this repo's `src/compact_v3`, which is my
from-scratch compact implementation of MLA, DeepSeekMoE and MTP.

The source workbook I am imitating is backed up at
a local backup outside this repository, kept out of version control because it is not mine to redistribute.
Read it by unzipping the OOXML with Python stdlib (`zipfile` + `xml.etree`); openpyxl and
pandas are not installed.

**All 14 sections are built and verified (308 checks). The workbook is complete.**

The read-through pass for clipped labels is done (2026-08-07): 14 long equation
labels were clipped by neighbouring blocks (sections 4, 5, 7, 8, 12, 13, 14) and
now sit above their blocks. What is left is a decision on whether to add the
deliberately excluded systems sections (node-limited routing, FP8 quantisation,
YaRN, DualPipe) as a second sheet.

## Files (all under `excel/`)

| File | Role |
|---|---|
| `build_sheet.ps1` | The builder. Drives Excel via COM and writes the xlsx. This is the source of truth; the workbook is generated, never hand-edited. |
| `verify.py` | Independent verification. Recomputes every block in pure-stdlib Python from the dump and asserts against what Excel produced. |
| `deepseekv3_by_hand.xlsx` | Output. |
| `_verify_dump.json` | Values dumped by the builder in one pass, read by `verify.py`. |
| `CAPABILITIES.md` | What this Excel supports and why. Read this before writing any formula. |

Run, from the repository root: `& .\excel\build_sheet.ps1` then `python excel\verify.py`.
Success is all three of:

- `ERROR CELLS: (none)`
- `DUPLICATE CELL WRITES: (none)`
- `PASSED all N checks`

All three matter. Each has caught a bug the others missed: error cells caught two
spill collisions, the duplicate-write scan caught a heading silently overwriting a
label, and the checks caught a dump address that had drifted from the builder.

## Hard constraints, each learned from a failed build

1. **No LAMBDA family.** This is Office Home & Student 2021 (perpetual, 16.0.20228).
   `LAMBDA`, `BYCOL`, `BYROW`, `MAP`, `REDUCE`, `MAKEARRAY`, `TOCOL`, `VSTACK`, `HSTACK`,
   `TAKE`, `CHOOSEROWS`, `TEXTSPLIT`, `GROUPBY`, `REGEXTEST` all return `#NAME?`. It is a
   licence limit, not a missing update, and it will never be fixed by updating Office.
   Available and confirmed working: `SEQUENCE` `RANDARRAY` `LET` `FILTER` `SORT` `SORTBY`
   `UNIQUE` `MMULT` `TRANSPOSE` `MINVERSE` `XMATCH` `MATCH` `INDEX` `LARGE` `SUMPRODUCT`
   `HYPERLINK` `UNICHAR`, spill refs (`E38#`), and array broadcast inside `IF`.
   `BYROW` is a trap: it returns `0` instead of `#NAME?`. Treat it as missing.
   Consequences: column-wise softmax is one `LET` per column; masks come from
   `IF(SEQUENCE(n,1)<=SEQUENCE(1,n),1,0)`; top-k must be built from `LARGE`/`MATCH`/`SORTBY`.

2. **Excel COM is mandatory.** Dynamic-array formulas need `<f t="array" ref=...>` plus
   `cm="1"` cell metadata and an `xl/metadata.xml` part. Only Excel writes that correctly.
   Use `.Formula2`, never `.Formula` (the latter applies implicit intersection).

3. **PowerShell caches the COM member binding per call site.** One helper cannot set both
   strings and numbers; whichever type goes first wins and the other throws
   `InvalidCastException`. Hence separate `Set-Text` and `Set-Num`. Keep them separate.

4. **`Dump-Range` must `return , $matrix`.** Without the unary comma PowerShell unrolls a
   single-row result and the caller's `[0]` indexes into the row instead of selecting it.

5. **Random inputs must be uniform on [-1,1]:** `=(RANDARRAY(r,c)-0.5)*2`, for both X and
   the projection weights. Plain `RANDARRAY` (all positive) saturates softmax to ~0.9 on one
   key; `RANDARRAY-0.5` (width 1) flattens it to uniform. Both collapse every output column
   to the same vector and destroy the teaching point. X in particular must be centred, or
   every token shares a mean direction that biases every dot product positive.

6. **`$xl.Calculation` must return to automatic before the last workbook closes**, or teardown
   throws.

7. **`$sectionCount` in the builder must equal the number of sections.** The table of
   contents writes one `HYPERLINK`/`LET` row per section; writing more rows than sections
   makes `INDIRECT` fail on an empty anchor and shows up as an error cell.

8. If the save reports a file lock, kill leftover Excel:
   `Get-Process EXCEL -ErrorAction SilentlyContinue | Stop-Process -Force`.
   Always check for strays at the end; my automation leaves them.

9. **Watch for spill collisions.** A block placed inside another block's spill range
   silently blocks it and cascades `#SPILL!` downstream. Section 12's proof block
   sat at row 800, inside both the b2 column (E797:E804) and the score2 block
   (O806:AL813), and took out five cells. Before placing a block, check the extent
   of every spill above and to the left of it.

10. **Keep dump addresses in step with the builder.** Section 12's dump still pointed
    at an earlier draft's layout, so it read empty cells and handed `verify.py` nulls.
    `verify.py` now fails fast with `EMPTY CELLS IN DUMP` and names the block, rather
    than throwing a `TypeError` hundreds of lines later.

11. **Assertions that depend on the random draw must be tested across draws**, not once.
    Two checks passed on the build I happened to run and failed roughly 1 draw in 10.
    Open the workbook, set `$xl.Calculation = -4135`, loop `CalculateFullRebuild()`
    20-30 times, and measure the invariant each time before trusting it.

## Layout conventions, reverse-engineered from his OOXML

- Canvas: column width 4.875, row height 24, Aptos Narrow 12, gridlines on, zoom 75.
- **Number format is General everywhere.** There is no rounding anywhere in his sheet; the
  narrow column does the rounding visually. Do not add number formats.
- Section header: text begins with `#` (this is load-bearing, the table of contents is a live
  `FILTER` over column A looking for it), 36pt bold, font colour `6970692`, row height 48,
  thick bottom border spanning the section's full width.
- Learned weights: orange fill `14083579` (Orange Accent 2, Lighter 80%).
- Computed activations: thin box outline via `Set-Box`, centred.
- Probabilities: red data bar `5920255`, fixed 0 to 1 scale.
- Labels are generated, not typed: `="q"&O24#`. Row labels right-aligned, column labels centred.
- Long equation labels go above their blocks, left-aligned. Right-aligned into the
  label column they overflow left, hit the neighbouring block and clip.
- Shapes are parameterised from a small config block (`seq`, `d_model`, `d_k`, `d_v`) and
  constants are derived, e.g. `SQRT(ROWS(O38#))` rather than `SQRT(4)`.
- Each section re-rolls its own inputs and stands alone, so a reader can jump to any section.
  This is his structure and it should be preserved.

## Row allocation

| Section | Header row |
|---|---|
| 1 `# Attention` | 21 |
| 2 `# Causal Attention and the KV Cache` | 60 |
| 3 `# Low-Rank Compression` | 108 |
| 4 `# Multi-head vs Latent Attention` | 145 |
| 5 `# The RoPE Conflict` | 195 |
| 6 `# Decoupled RoPE` | 250 |
| 7 `# Weight Absorption and the Cache Ledger` | 340 |
| 8 `# SwiGLU and Top-k MoE` | 455 |
| 9 `# DeepSeekMoE: Fine-Grained and Shared Experts` | 510 |
| 10 `# Sigmoid Affinity and Route Scale` | 580 |
| 11 `# Load Collapse` | 632 |
| 12 `# Auxiliary-Loss-Free Load Balancing` | 730, content to 841 |
| 13 `# The V3 Block and the Sequence-Wise Balance Loss` | 860, content to 996 |
| 14 `# Multi-Token Prediction` | 1010, content to 1207 |

Rows 5-18 are reserved for the table of contents as it grows to 14 entries.

Sections 11 and 12 are deliberately coupled: 11 forces a collapse with a fixed
`advantage` (cell F639, currently 3.0) applied to experts 2, 5 and 8, and 12 reads
the same number through `F738 = F639` and shows the bias undoing it. Changing the
advantage in one place changes both, which is intended. `update_rate` (F737, 0.75)
must satisfy `6 * rate > advantage`, because three sign-based steps open a bias gap
of at most `6 * rate` and it has to clear the advantage to move any tokens.

## Hand-scale config, fixed for all 14 sections

```
d_model 8, n_heads 2, seq 6
q_lora_rank 4, kv_lora_rank 3
qk_nope_head_dim 3, qk_rope_head_dim 2, v_head_dim 3
n_routed_experts 8, n_shared_experts 1, top_k 2, expert_hidden 6
```

Chosen so the section 7 cache ledger reads MHA 16 floats per token against MLA 5, a 3.2x cut
visible without a calculator, mirroring `cache_report()` in `src/compact_v3/mla.py`.

## The remaining table of contents

Each section is the smallest possible edit of the one before it. That chain is the whole
design; do not reorder it.

| # | Section | Delta | Content |
|---|---|---|---|
| 3 | Low-Rank Compression | drops attention for one screen | `c = W_down x`, `x_hat = W_up c`, reconstruction error beside the original. The LoRA trick alone. |
| 4 | Multi-head Latent Attention | puts 3 inside 2 | cache only `c_KV` (rank 3), rebuild K and V per head at read time |
| 5 | The RoPE Conflict | one counterexample | show numerically that `RoPE(W_UK c) != W_UK RoPE(c)`. Two cells that should match and do not. The reason section 6 exists. |
| 6 | Decoupled RoPE | split the head | `q = [q_content ; q_rope]`, score `= q_c.k_c + q_r.k_r`, one shared rope key head fed from x. Mirrors `MultiHeadLatentAttention.reference_manual`. |
| 7 | Weight Absorption and the Cache Ledger | reassociate the matmul | fold `W_UK` into the query and `W_UV` after attention, side by side with 6, identical output. Ends with MHA vs GQA vs MLA floats per token. Mirrors `decode` vs `decode_naive`. |
| 8 | SwiGLU and Top-k MoE | new half | dense SwiGLU expert, softmax router picking 1 of 8 |
| 9 | DeepSeekMoE: Fine-Grained and Shared Experts | split and add | quartered experts, higher top-k, one always-on shared expert |
| 10 | Sigmoid Affinity and Route Scale | swap the gate | per-expert sigmoid, top-k, renormalise over selected only, times `route_scale`. V3's change from V2. Mirrors `TopKRouter`. |
| 11 | Load Collapse | a failure, no new mechanism | route 24 tokens through 8 experts, count load, compute normalised entropy, watch three experts take everything |
| 12 | Auxiliary-Loss-Free Load Balancing | one vector, one place | bias `b` added to the selection score but never to the gate weight. Three update steps moving load toward uniform, plus a proof cell that output weights are untouched by `b`. Mirrors `LoadBalancer`. |
| 13 | The V3 Block and the Sequence-Wise Balance Loss | assemble | RMSNorm, residual, MLA, MoE, dense first layer then MoE layers |
| 14 | Multi-Token Prediction | one module off the last hidden state | `RMSNorm(h_i)` and `RMSNorm(Emb(t_{i+1}))` concatenated, merged by `M`, one transformer block, shared output head, loss on `t_{i+2}` at weight lambda; then the same module as a speculative draft head. Mirrors `MTPObjective`. |

Sections 5, 7, 11 and 12 are the ones that earn the workbook its existence: each is a claim
about two numbers being equal or unequal, which a spreadsheet proves better than prose.

Deliberately excluded, and not to be added without asking: node-limited routing, FP8
fine-grained quantisation, YaRN context extension, DualPipe. They are training and systems
mechanisms, not forward-pass mechanisms, and none is in `src/compact_v3`.

## Workflow for each new section

1. Read `excel/CAPABILITIES.md` and the existing section 2 block in `build_sheet.ps1` to
   match style.
2. Add the section block to `build_sheet.ps1`; bump `$sectionCount`.
3. Add the section's blocks to the `$payload` dump.
4. Add **independent** recomputation to `verify.py`. Recompute from first principles in pure
   stdlib Python; never assert a value by reading it back from the same formula.
5. Build, then verify. Both must be clean: `ERROR CELLS: (none)` and `PASSED all N checks`.
6. Check invariants hold across several forced recalcs, since `RANDARRAY` re-rolls on every
   F9 and the reader will press it. Set `$xl.Calculation = -4135`, loop `CalculateFullRebuild()`,
   read the block each time.
7. Optionally export a PNG to eyeball layout: Excel `Visible = $true`, `Range.CopyPicture(1,2)`,
   `ChartObjects().Add(...)`, `Chart.Paste()`, `Chart.Export(path,'PNG')`. Needs Excel visible
   and a ~500ms sleep after each clipboard step, and do not call `ChartArea.Clear()`.

## House rules

- Never add a `Co-Authored-By` trailer to a commit.
- In prose (docs, comments, commit text): no em dashes, no emojis.
- Build the simplest thing that works; no speculative abstraction.
- Ground every progress claim in an actual tool result from the session.
- Nothing has been committed yet. Current branch is `mlops-experiments`; `excel/` is untracked.

## Two conventions section 13 added

- **`Set-Sci`** applies a `0.0E+00` number format. It is the only number formatting in
  the workbook and exists because General renders 1e-6 as a bare `0` in a 4.875-wide
  column. Used on `rms_eps`, `balance_coef` and the balance loss only. Do not spread it.
- **PowerShell hash keys are case-insensitive**, so a dump cannot hold both `V` and `v`.
  Section 13 names the FFN one `ffn_v`.

## Two traps section 14 hit, worth knowing before editing any section

- **A spill reference reads only the formula it is anchored to.** Section 14 builds the
  embedding lookup as four per-column formulas, so `O1025#` is 8x1, not the 8x4 block.
  Using it made `rms_e` a scalar, left the concat block one column wide, and cascaded
  `#SPILL!` through 28 cells. Multi-column blocks built one column at a time must be
  read as a plain range (`O1025:R1032`).
- **A demonstration can be flaky even when the arithmetic is exact.** The
  degenerate-objective panel showed `E^T e_target` peaking somewhere other than the
  target on about 1 draw in 40, because a longer off-target embedding column can
  out-dot the target's own squared norm. Normalising the embedding columns to unit
  length makes the diagonal exactly 1.0 and every off-diagonal a cosine below it, so
  the claim holds on every recalc. Measured 40/40 after the change.

## Reference: what section 14 contains

- The merge: `merge(concat(RMSNorm(h_i), RMSNorm(Emb(t_{i+1}))))`, a `2*d_model ->
  d_model` projection (`mtp.py:63`), then one transformer block, then the shared
  output head over `final_norm`, giving logits for `t_{i+2}`.
- `combined_loss = main_loss + lambda * mtp_loss`, lambda annealing 0.3 to 0.1 at
  67.6% of training (`mtp_weight_schedule`).
- **The alignment is the section's real payload and the easy thing to get silently
  wrong.** The module is fed `t_{i+1}` and predicts `t_{i+2}`. Feed it the target
  itself and it reads the answer off its own input through the tied output head and
  learns the identity map; see `experiments/GATE_U_MTP_OBJECTIVE.md` and the docstring
  on `MTPObjective.forward`. Lay `h_i`, `input_tokens` and `future_targets` out as
  three offset rows so the off-by-one is visible rather than argued.
- Close with the same module as a speculative draft head: because it consumes
  `t_{i+1}`, which the main model just emitted, a short accept/reject trace works.
