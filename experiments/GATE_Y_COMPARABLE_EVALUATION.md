# Gate Y — comparable evaluation and a shippable artifact

Status: experimental, on branch `mlops-experiments`. `src/compact_v3/`, `v3_cli.py`, `serve.py`, `complete.py`
and `tests/` are byte-identical to `master`. Only `pyproject.toml`, `uv.lock` and `.gitignore` changed.

## Specification

Gate X built instrumentation. This runs the loop end to end and asks two questions the roadmap already poses:

1. **Is 41.35 comparable to anything?** The README converts it by hand: "at a measured fertility of 1.13 tokens
   per word, 41.35 subword perplexity is 67.1 word-level, against 29.4 for a published 6-layer 156M-parameter
   decoder", concluding the model is "roughly 2.3x off". That conversion has never been checked against the
   metric the literature actually reports.
2. **What does int8 cost?** The roadmap: "int8 with a scale per 32 values is about 165MB, so the perplexity cost
   needs measuring against the 41.35 baseline before anything ships."

## Implementation

`mlops/hf_wrapper.py` gives `CompactV3Model` a `PreTrainedModel` face. `mlops/eval_harness.py` drives lm-eval
through it. `mlops/quantize_int8.py` measures quantization against the repo's own `evaluate_tokens`.

The dependency cost is real and is the first finding. `transformers` 5.x requires `tokenizers<=0.23.0`; this repo
pinned `tokenizers>=0.23.1`. The only `transformers` release satisfying that pin is **4.17.0, from 2022**, which
also drags `huggingface-hub` from 1.26 to 0.36. The floor was relaxed to `>=0.22.0` and the 90 tests pass
unchanged on 0.22.2. The eval stack lives in a `[dependency-groups] eval` block so `uv sync` still yields the
lean training environment.

## Result 1: the headline number is not comparable, and not for the reason the README assumes

lm-eval, `wikitext` task, all 62 test documents, through the wrapper:

```text
word_perplexity   1028.95
byte_perplexity      3.659
bits_per_byte        1.871
```

Against the README's hand-derived 67.1. The gap is not arithmetic. Working backwards:

| what was measured | subword PPL | fertility | note |
|---|---:|---:|---|
| repo `evaluate_tokens`, `data_v3_103` validation | **41.35** | | reproduces the headline exactly, 470 windows |
| same split, continuous text, per-line `[EOS]` removed | 52.00 | | |
| WikiText-103 **test**, raw, continuous | 60.84 | 1.135 | |
| WikiText-103 test, **detokenized** (what lm-eval feeds) | **402.56** | 1.371 | |

The cause is `lm_eval/tasks/wikitext/preprocess_wikitext.py`. lm-eval applies `wikitext_detokenizer` before
scoring: ` @-@ ` becomes `-`, ` @.@ ` becomes `.`, ` . ` becomes `. `, `= =` becomes `==`, ` N ` becomes ` 1 `.
It then counts words on the *original, pre-detokenization* document, which the file flags in a comment.

This model was trained on `wikitext-103-raw-v1` with those artifacts intact. Detokenized text is therefore out
of distribution for it, and 6.6x worse in subword perplexity (60.84 to 402.56). Its own tokenizer, fit on raw
text, also fragments the detokenized form more: fertility rises from 1.135 to 1.371.

Published WikiText word-level perplexities are computed on the detokenized convention. So the 29.4 baseline in
the README is measured on text this model never saw. **The "roughly 2.3x off" claim is not supportable as
written**, in either direction: the comparison is between two different corpora, not two models.

The three contributions, separated: per-line `[EOS]` injection accounts for 41.35 to 52.00 (26%), the
validation-to-test split for 52.00 to 60.84 (17%), and detokenization for 60.84 to 402.56 (562%).

### What this does not say

It does not say the model is worse than believed. 41.35 is a correct measurement of what it measures. It says
that number answers a different question than the published baselines answer, and that the bridge between them
is not a fertility exponent.

### What would fix it

Either train on detokenized WikiText, which is what the literature does and would make `word_perplexity`
directly comparable, or find a baseline evaluated on raw WikiText. The first is a real change to `data.py` and
invalidates cross-gate comparisons; the second may not exist. This gate does not choose.

## Result 2: int8 is nearly free, and the roadmap's group size is the worse choice

Both rows measured against the repo's own deterministic `evaluate_tokens` over all 470 validation windows, in
the same process as the fp32 baseline.

| config | fp32 PPL | int8 PPL | delta | fp32 MB | int8 MB | reduction |
|---|---:|---:|---:|---:|---:|---:|
| per-row (torchao default) | 41.3537 | 41.3593 | **+0.014%** | 621.5 | **208.6** | 66.4% |
| per-group 32 (roadmap spec) | 41.3537 | 41.3647 | +0.027% | 621.5 | 228.8 | 63.2% |

Per-row wins on both axes: half the perplexity cost and 20MB smaller, because per-group-32 stores 16x more
scales than per-row for no accuracy benefit at this size. The roadmap's "scale per 32 values" should be
per-row. This is the same shape of result as Gate P: a value borrowed from a larger-scale setting measured
worse here.

At 0.014% on 41.35, int8 is free. The 165MB estimate in the roadmap was optimistic; the measured figure is
208.6MB of parameters.

Latency was not measured. torchao's int8 weight-only kernel relies on `torch.compile` for its speedup, so a
wall-clock number without compile would understate it.

## Result 3: three tools, three failures, one root cause

`model.py:33` ties the output head to the embedding, and `MTPObjective` holds both by reference, so three names
alias one storage. Every ecosystem tool tried in this gate broke on it, differently:

| tool | failure | visible? |
|---|---|---|
| `safetensors.save_file` | refuses aliased tensors outright | yes, loud error |
| `transformers.save_pretrained` / `from_pretrained` | saved fine, reloaded wrong | **no** |
| `torchao.quantize_` | `NotImplementedError` in `aten.view` on the second visit to the same tensor | yes, loud error |

The middle row is the dangerous one and is the most important finding in this gate. transformers 5.x reads tie
information from two different places:

- `save_pretrained` drops aliases via `_get_tied_weight_keys`, which reads `_tied_weights_keys` directly.
- `from_pretrained` restores them via `get_expanded_tied_weights_keys`, which begins
  `getattr(self.config, "tie_word_embeddings", False)` and **returns an empty map** when that attribute is
  absent.

A config without `tie_word_embeddings` therefore saves a correct, smaller file and reloads a model with two
randomly initialized matrices, with a clean load report and no error.

Two further transformers 5.x traps, both of which produce misleading errors:

- `_tied_weights_keys` is a **dict** (`{alias: source}`) in 5.x; the still-current 4.x documentation shows a
  list, and passing a list fails with `AttributeError: 'list' object has no attribute 'keys'` deep inside
  `save_pretrained`.
- Omitting `self.post_init()` fails at load with `'CompactV3ForCausalLM' object has no attribute
  'all_tied_weights_keys'`.

## Result 4: non-persistent buffers are silently destroyed by from_pretrained

`rope.py:19-20` registers `cos` and `sin` with `persistent=False`. They are therefore in no state_dict and in no
safetensors file. transformers builds the module on the meta device and materializes it, and because nothing in
the checkpoint claims those names, they are never filled.

They come back as **uninitialized memory**, not zeros:

```text
correct cos[0,:6]: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
loaded  cos[0,:6]: [-483.25, 2.11e-42, 0.0, 0.0, 0.0, 0.0]
```

All 593 parameters loaded correctly. The load report was clean. No error, no warning. The model produced
confident output with a max logit error of 10.4 against the source.

This was caught only because the export was verified by comparing logits. A size check would have passed it, a
parameter-by-parameter diff would have passed it (all 601 state_dict entries were bit-identical), and a
config diff would have passed it. `mlops/hf_wrapper.py` now rebuilds the tables after load.

## Final artifact

With the tie map declared and the buffers rebuilt:

```text
compact_v3_wikitext103_2ep.pt   1,864,088,279 bytes
  -> model.safetensors            621,158,408
     tokenizer.json                 2,279,541
     config.json + tokenizer_config      1,189
  total                           623,444,694    reduction 66.6%
  round-trip bit-identical: true
```

## Open issues

- The detokenization finding needs a decision, and it changes what the README can claim.
- W&B ran offline only. No account was created and nothing was synced.
- `mlops/export.py` and `mlops/verify_export.py` from Gate X are now redundant; `save_pretrained` does the same
  job correctly once the tie map is declared.
- int8 latency unmeasured, pending a `torch.compile` run.
- The wrapper's `forward` ignores `attention_mask` and returns no `past_key_values`, so it is correct for
  scoring and not yet for batched generation through `generate()`.
