# Gate S — Tokenizer decoder fix and first qualitative text assessment

## Status

Completed on 2026-08-04. `complete.py` added as a permanent text-in/text-out tool.

## Motivation

Every generation test through Gate R inspected raw token IDs only (`"generated_tokens": [203, 227, 200, ...]`).
No gate had ever decoded output back to readable text, and no gate had ever run the model on a real, human-typed
prompt — every prior generation smoke used `torch.randint(...)` for the prompt. Asked directly when the model
would be able to complete a sentence like "The cat sat on", this gap needed to be closed with a real answer, not
a perplexity number alone.

## A blocking bug, found immediately

The first decode attempt printed literal BPE artifacts instead of text: `'T he Ġcat Ġsat Ġon Ġthe Ġcoast ...'`.
`data_v3.py`'s `_train_tokenizer()` sets `tokenizer.pre_tokenizer = ByteLevel(...)` but never set a matching
`tokenizer.decoder` — so `.decode()` fell back to naively joining raw tokens instead of reversing the byte-level
encoding back to text. This affected every tokenizer ever trained by this project (cached `data_v3/tokenizer.json`
included) and would have blocked readable output regardless of model quality.

## Fix

- `data_v3.py`: `_train_tokenizer()` now sets `tokenizer.decoder = ByteLevelDecoder()` before training/saving, so
  newly-built tokenizers decode correctly. `load_tokenizer()` now always attaches the decoder after loading,
  so the existing cached `tokenizer.json` (trained before this fix) decodes correctly too, with no need to
  retrain it or invalidate its recorded SHA256 provenance hash.
- Added `test_tokenizer_decode_round_trips_to_readable_text` (`tests/test_data_v3.py`): encodes and decodes a
  sentence through both a freshly-trained and a reloaded-from-disk tokenizer, asserts exact round-trip text.

## `complete.py` — new permanent tool

A small CLI: load a checkpoint and tokenizer, encode a real text prompt, run `generate_cached`, decode the
result, print JSON with `prompt`, `completion`, and `continuation_only`. Added
`test_complete_cli_produces_readable_text` (`tests/test_v3_cli.py`) exercising it end-to-end on a tiny checkpoint.
Full active suite: 65 passed.

## What the current best checkpoint actually produces

Using Gate Q's best checkpoint (`..._gateq_mtp_anneal.pt`, 292.54 validation PPL, step 1954), greedy decoding:

```text
prompt:      "The cat sat on"
completion:  "The cat sat on the coast of the United States . The storm was also used as a tropical storm . \n ="

prompt:      "The history of the city began"
completion:  "The history of the city began in the United States . The city was built in the United States
              and was the first to be the"
```

## Interpretation

The model has learned surface English grammar from WikiText-2 — correct capitalization, punctuation, article
usage, subject-verb agreement — and even reproduces Wikipedia-specific structural conventions (the trailing `=`
in the first example mimics WikiText's `= Section Header =` markup). It has **not** learned topical coherence:
"The cat sat on" continues into unrelated geography/weather content, "United States" is reused within a few
tokens in both examples, and the second completion trails off mid-phrase. This is consistent with a validation
perplexity of 292 — well-converged small language models on comparable data typically reach perplexities in the
20-40 range, roughly an order of magnitude lower than where this checkpoint currently sits.

Two separate reasons this isn't "The cat sat on the mat"-quality yet:

1. **Undertrained.** Every real-corpus run in this project's history trained a checkpoint from scratch for at
   most ~4M tokens (Gates O-Q), against a ~2.28M-token training set — under two epochs. No single checkpoint has
   been trained substantially longer than that. This is a legitimate next experiment: keep training the same
   architecture much further and see how far perplexity actually drops on this data before plateauing.
2. **Domain-limited even if fully converged.** WikiText-2 is Wikipedia article text. A phrase like "the cat sat
   on the mat" is not a natural Wikipedia continuation regardless of training quality — a fully converged
   WikiText-2 model would still tend to continue "The cat sat on" with encyclopedia-flavored content, just more
   coherently than it does now. General-purpose, any-sentence completions would need a larger, more diverse
   training corpus, not just more steps on WikiText-2.

## Validation

```text
Tokenizer decode test: 1 passed
complete.py CLI test:  1 passed
Active V3 suite:       65 passed
Real checkpoint demo:  2 prompts, readable output confirmed
```

## Next gate

No architecture change implied by this gate. Two independent follow-ups are now well-motivated by direct
evidence rather than perplexity alone: (1) a longer training run on the current WikiText-2 setup to measure how
much headroom remains before this architecture plateaus on this data; (2) if general-purpose completions are the
goal, evaluate a larger/more diverse corpus, with the same provenance diligence WikiText-2 got.
