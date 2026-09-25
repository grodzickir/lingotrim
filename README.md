# lingotrim

**Trim multilingual LLMs down to the languages you actually use — losslessly and verifiably.**

Modern open models ship with huge multilingual vocabularies (Qwen3.8: 248k tokens), but most
deployments only ever see one or two languages plus code. `lingotrim` removes the tokens you never
use from the tokenizer, the input embeddings and the LM head, and then **proves** the trimmed model
behaves identically on your target languages.

```text
$ lingotrim trim Qwen/Qwen3-0.6B qwen3-0.6b-enpl --corpus corpus/en.txt corpus/pl.txt corpus/code.txt
vocab: 151,936 -> 98,000 (-35.5%)
weights: 1.40 GB -> 1.19 GB, 110.5M parameters removed

$ lingotrim verify Qwen/Qwen3-0.6B qwen3-0.6b-enpl --corpus heldout/ --logits
tokenization: LOSSLESS on 4,813 texts / 849,245 tokens
  vocabulary coverage: 99.9580% of tokens kept
logits: max |diff| on kept tokens = 0.00e+00
```

## Why it is lossless

1. **Token selection** — keep every token seen in a target-language corpus, all pure-ASCII tokens
   (code, English), all special tokens and optionally whole Unicode scripts (`--scripts Latin`).
2. **BPE merge ancestry closure** — BPE builds tokens from intermediate tokens. If a kept token is
   produced by `a + b`, both `a` and `b` are kept too, recursively. This guarantees that any text
   whose original tokenization uses only kept tokens is tokenized *identically* after trimming.
3. **Byte fallback stays intact** — the 256 byte-level base tokens are always kept, so text in a
   removed language still encodes and decodes correctly; it just costs more tokens.
4. **Row slicing, not retraining** — the kept rows of every vocab-sized tensor are copied verbatim,
   so logits for kept tokens are bit-identical. Transformer layers are untouched.

Weight slicing works directly on `safetensors` bytes (memory-mapped, streamed shard by shard, any
dtype including BF16/FP8), so trimming a 27B model needs neither a GPU nor torch.

## Real numbers

| Model | Target | Vocab | Removed params | Size impact |
|---|---|---|---|---|
| Qwen3-0.6B | EN + PL + code | 151,936 → 98,000 (−35.5%) | 110.5M | −15% (1.40 → 1.19 GB) |
| Qwen3.8-27B | EN + PL + code | 248,320 → 132,143 (−46.8%) | ~595M | ~−2% (≈1.2 GB of BF16) |

Token counts after trimming (Qwen3-0.6B): Polish 24 → 24, German 13 → 16, Russian 16 → 33,
Chinese 7 → 14 — target languages are unaffected, removed languages fall back to bytes.

**Honest take:** vocabulary trimming is a big win for small models (and for embedding/LM-head
bound workloads like large-batch prompt processing), but for 27B+ models the embeddings are only a
few percent of the weights. Removing a language's *knowledge* from the transformer layers requires
lossy structural pruning plus healing — see the roadmap.

## Install

```bash
uv tool install git+https://github.com/<you>/lingotrim            # core (numpy, tokenizers, regex)
uv tool install "lingotrim[corpus,verify] @ git+https://github.com/<you>/lingotrim"
```

## Usage

```bash
# 1. What is in the vocabulary?
lingotrim inspect path/to/model

# 2. Get language samples (FineWeb / FineWeb-2 / code); or bring your own text files
lingotrim corpus en pl code --out corpus --chars 5000000
lingotrim corpus mydomain=my-org/my-dataset:config@text --out corpus

# 3. Trim
lingotrim trim path/to/model path/to/model-enpl --corpus corpus/

# 4. Verify on held-out text (use --logits for small models; needs the `verify` extra)
lingotrim verify path/to/model path/to/model-enpl --corpus heldout/ --logits
```

Every run writes `lingotrim_report.json` with vocabulary sizes, reasons each token was kept, removed
parameters and bytes saved.

## Supported models

Byte-level BPE tokenizers in Hugging Face `tokenizer.json` format with `safetensors` weights: Qwen
2/2.5/3/3.x, Llama 3.x, GPT-style models and most of their fine-tunes. Token ids referenced in
`config.json` / `generation_config.json` (`*_token_id`, including nested `text_config`) and in
`tokenizer_config.json` are remapped automatically.

Not yet supported: SentencePiece-style tokenizers (Gemma, Mistral v1/v2, Llama 2), models with
block-quantized embeddings (scale tensors not sliced).

## Roadmap

The full "specialize a general model for my languages and my domain" pipeline:

| Stage | What | Lossless | Status |
|---|---|---|---|
| 1. Vocabulary trimming | tokenizer + embeddings + LM head | yes | ✅ done |
| 2. GGUF export | llama.cpp conversion of the trimmed model (register tokenizer hash) | yes | planned |
| 3. Structural pruning | drop FFN channels / heads / layers least active on target-language + code calibration data (Compact, Minitron-style width pruning) | no | planned |
| 4. Healing & specialization | distill from the original model on target-language + domain (e.g. code) data, LoRA or full FT | — | planned |
| 5. Quantization | GGUF k-quants / QAT / ternary | no | planned |
| 6. Eval harness | per-language perplexity + downstream (e.g. HumanEval, Polish benchmarks) before/after each stage | — | planned |

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
```

## License

MIT
