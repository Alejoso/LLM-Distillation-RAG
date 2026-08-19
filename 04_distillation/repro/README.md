# Repro: student degeneration (EOS never learned)

Minimal, reproducible demonstration of the root cause behind the degenerate
student outputs (long English repetition loops answering short Spanish legal
questions, e.g. 70-93% degenerate answers in the 1500-item eval).

## Root cause

In `DistillationDataset` the `response_mask` weighted only non-pad tokens, and
because `pad_token == eos_token` the EOS position always received **zero loss
weight** in both the soft (KD) and hard (CE) terms. The student learned the short
Spanish answer but was never trained to emit EOS, so at inference it never
stopped: it produced the answer and then rambled to the token cap, drifting into
the base model's repetitive English text.

## The fix (in `../pipeline_destilacion_apolo.py`)

- `DistillationDataset`: add `eos_mask` marking the first EOS/pad position after
  the response.
- `train_student`: apply the hard-CE loss on that EOS position (KD skips it,
  there are no teacher logits there). Reuses existing teacher logits, so only the
  student training needs re-running, not the expensive teacher inference. The same
  `eos_mask` also makes a full data regeneration correct, with no reliance on the
  tokenizer parsing a literal `</s>` string — both plans behave identically.
- `generate_model_response`: greedy (reproducible) + `repetition_penalty=1.3` +
  `no_repeat_ngram_size=3` + explicit `eos_token_id`.

## How to run (isolated, pinned environment)

```bash
python3.12 -m venv venv           # base interpreter must be >= 3.12
./venv/bin/pip install -r requirements-repro.txt
./venv/bin/python repro_eos.py    # downloads base TinyLlama once (~2.2 GB)
```

Runs on Apple MPS or CPU. It micro-distills base TinyLlama on a few real
Spanish QA pairs from `../data/raw_train.json`, twice — with the buggy masking
(EOS excluded) and with the fix (EOS trained) — then generates greedily.

## Result

| Probe | Buggy masking | Fixed (EOS) greedy | Fixed + eval decoding |
|-------|---------------|--------------------|-----------------------|
| Art. 2       | loops, 100 tok | stops, 6 tok  | stops, 6 tok |
| Building     | loops, 100 tok | loops, 100 tok | stops, 8 tok |
| Held-out     | loops, 100 tok | stops, 6 tok  | stops, 6 tok |

The only difference between "buggy" and "fixed" is the `eos_mask`. It flips
"never stops" into "stops at EOS". The eval decoding config is the second layer
that cuts any residual loop (the "building" probe).

Note: this is a mechanism proof on the base model (CE, no teacher logits), not
the full 76k-item distillation run. The decisive validation of the fix at scale
is a student-training re-run on Apolo reusing the existing teacher logits.
