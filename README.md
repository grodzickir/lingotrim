# lingotrim

**Trim multilingual LLMs down to the languages you actually use — losslessly, verifiably, in one command.**

Open models ship with huge multilingual vocabularies (Qwen3.8: 248k tokens across 28 scripts), but
most people use English plus one language, plus code. `lingotrim` keeps exactly those, removes the
rest from the tokenizer, embeddings and LM head, **proves** the trimmed model is identical on your
languages, and exports a quantized GGUF you can run in Ollama.

```bash
npx lingotrim run Qwen/Qwen3-0.6B --languages en,pl --quantize Q4_K_M --ollama qwen3-0.6b-enpl
# or, with uv:  uvx --from "lingotrim[corpus,gguf]" lingotrim run ...
```

```text
== 1/6 model      downloading Qwen/Qwen3-0.6B
== 2/6 corpus     en / pl / code samples (FineWeb-Edu, FineWeb-2, CodeParrot)
== 3/6 trim       vocab 151,936 -> 97,323 (-35.9%), 112M parameters removed
== 4/6 verify     tokenization LOSSLESS, coverage 99.91% on 1,542 held-out texts
== 5/6 gguf       pre-tokenizer: qwen2 -> trimmed-en-pl-code-bf16.gguf
== 6/6 quantize   Q4_K_M 382 MiB (5.01 BPW) -> ollama model: qwen3-0.6b-enpl
```

```text
$ ollama run qwen3-0.6b-enpl "Odpowiedz krótko po polsku: ile to jest 7 razy 8?"
7 razy 8 to 56.
```

Any language in the table below works the same way: `--languages en,de`, `--languages ja`,
`--languages en,uk,pl`. English is always kept (it dominates instruction tuning); code is kept
unless you pass `--no-code`.

## Why it is lossless

1. **Token selection** — keep every token that appears when tokenizing a sample of your languages,
   all pure-ASCII tokens (code, English), all special tokens, and optionally whole Unicode scripts.
2. **BPE merge ancestry closure** — if a kept token is built by merging `a + b`, both parts are kept
   too, recursively. Any text whose original tokenization used only kept tokens is tokenized
   *identically* after trimming.
3. **Byte fallback stays intact** — the 256 byte-level base tokens are always kept, so removed
   languages still encode and decode correctly, just with more tokens.
4. **Row slicing, not retraining** — kept rows of every vocab-sized tensor are copied verbatim, so
   logits for kept tokens are bit-identical. Transformer layers are untouched.
5. **Verification is part of the pipeline** — held-out text is tokenized with both tokenizers and
   compared; the run aborts before export if anything differs. `--verify-logits` additionally
   compares model outputs (small models).

Weight slicing works directly on `safetensors` bytes (memory-mapped, streamed shard by shard, any
dtype including BF16/FP8), so trimming a 27B model needs neither a GPU nor torch.

## Results

| Model | Languages | Vocabulary | Removed params | Weights |
|---|---|---|---|---|
| Qwen3-0.6B | en + pl + code | 151,936 → 97,323 (−35.9%) | 112M | 1.40 → 1.19 GB (−15%) |
| Qwen3.8-27B | en + pl + code | 248,320 → 132,143 (−46.8%) | ~595M | ≈ −1.2 GB of BF16 (−2%) |

Token counts after trimming (Qwen3-0.6B): Polish 24 → 24, German 13 → 16, Russian 16 → 33,
Chinese 7 → 14 — target languages unaffected, removed languages fall back to bytes.

**Honest take:** vocabulary trimming is a big win for small models and for embedding/LM-head-bound
workloads (large-batch prompt processing, speculative decoding drafts). For 27B+ models the
embeddings are only a few percent of the weights; removing a language's *knowledge* from the
transformer layers needs lossy structural pruning plus healing — see the roadmap.

## Install

```bash
npx lingotrim --help                                   # Node 18+, no Python needed (uv installs it)
uv tool install "lingotrim[corpus,gguf]"               # or as a uv tool
pip install "lingotrim[corpus,gguf]"                   # or plain pip
```

Extras: `corpus` (Hugging Face `datasets` for language samples), `gguf` (torch CPU + transformers
for `convert_hf_to_gguf.py`), `verify` (logit comparison). The core trim/verify path needs only
`numpy`, `tokenizers` and `regex`.

`llama.cpp` (converter + prebuilt `llama-quantize` for Windows, Linux, macOS) is downloaded on
first use into `~/.cache/lingotrim` (`%LOCALAPPDATA%\lingotrim` on Windows). Point `LLAMA_CPP_DIR`
or `--llama-cpp` at your own checkout/build to skip that.

## Usage

```bash
# Everything in one go (idempotent: rerun to resume after a failure)
lingotrim run Qwen/Qwen3.8-27B --languages en,pl --out ./qwen3.8-enpl \
    --quantize Q4_K_M Q5_K_M --ollama qwen3.8-27b-enpl

# Local model dir, your own corpus, no GGUF
lingotrim run ./my-model -l en,de --source legal=my-org/legal-corpus@body --out ./out

# Individual steps
lingotrim inspect ./model                                 # vocabulary by Unicode script
lingotrim corpus en pl code --out corpus --chars 20000000 # writes corpus/fit and corpus/heldout
lingotrim trim ./model ./model-enpl --corpus corpus/fit
lingotrim verify ./model ./model-enpl --corpus corpus/heldout --logits
lingotrim export ./model ./model-enpl --out model-enpl.gguf --quantize Q4_K_M --ollama my-model
```

`run` writes `run-<languages>.json` (summary) and `lingotrim_report.json` (why each token was
kept, sliced tensors, bytes saved) into the output directory.

### Languages

`en pl de fr es it pt nl sv da no fi cs sk hu ro hr sl lt lv et tr vi id ms ca uk ru bg sr el he
ar fa hi bn ta th zh ja ko` — plus any raw FineWeb-2 config such as `pol_Latn`, and custom
sources via `--source name=dataset[:config][@column]`.

## Supported models

Byte-level BPE tokenizers in Hugging Face `tokenizer.json` format with `safetensors` weights: Qwen
2 / 2.5 / 3 / 3.x, Llama 3.x, GPT-style models and their fine-tunes. Token ids referenced in
`config.json`, `generation_config.json` (including nested `text_config`) and `tokenizer_config.json`
are remapped automatically. GGUF export pins the original pre-tokenizer type, which llama.cpp's
converter would otherwise fail to detect for a trimmed vocabulary.

Not yet supported: SentencePiece tokenizers (Gemma, Mistral, Llama 2), block-quantized checkpoints
(scale tensors are not sliced), vision projectors (`--mmproj`) — the text model is exported.

## Roadmap

| Stage | What | Lossless | Status |
|---|---|---|---|
| 1. Vocabulary trimming | tokenizer + embeddings + LM head | yes | done |
| 2. GGUF export + quantization + Ollama | llama.cpp converter and k-quants | yes / quant | done |
| 3. npm / npx distribution | `npx lingotrim` via uv | — | done |
| 4. Structural pruning | drop FFN channels / heads / layers least active on target-language + code calibration data | no | planned |
| 5. Healing & specialization | distill from the original on target-language + domain (e.g. code) data | — | planned |
| 6. Eval harness | per-language perplexity + downstream benchmarks before/after each stage | — | planned |

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
```

## License

MIT
