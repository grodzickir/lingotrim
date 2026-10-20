# lingotrim

**Trim a multilingual LLM's vocabulary down to the languages you use (tokenizer, embeddings and LM head),
with a tokenization check on held-out text and optional GGUF export for Ollama.**

Open models ship with large multilingual vocabularies (Qwen3.8-27B: 248,320 tokens), but many users need
English, one more language and code. `lingotrim` keeps the tokens those need (plus everything required to
keep byte-level BPE tokenization consistent), removes the rest from the tokenizer, the input embedding
and the LM head, checks on held-out text that the trimmed tokenizer reproduces the original tokenization,
and can export a quantized GGUF and import it into Ollama.

```bash
uvx --from "lingotrim[corpus,gguf] @ git+https://github.com/grodzickir/lingotrim" \
    lingotrim run Qwen/Qwen3-0.6B --languages en,pl --quantize Q4_K_M --ollama qwen3-0.6b-enpl
```

Example of the progress output (illustrative: the numbers come from the Qwen3.8-27B run in
[Results](#results), and the lines use the format the tool prints):

```text
== 3/6 trim
vocab 248,320 -> 135,657 (-45.4%), 1,154M parameters removed
== 4/6 verify
tokenization preserved, token coverage 99.9153% on 9,341 held-out texts (N texts with removed tokens were skipped)
```

`--languages en,de`, `--languages ja`, `--languages en,uk,pl` work the same way. English is always kept;
code is kept unless you pass `--no-code`.

## How it works

1. **Token selection**: keep every token that appears when tokenizing a sample of your languages, all
   pure-ASCII tokens (code, English), all special tokens, and optionally whole Unicode scripts.
2. **BPE merge ancestry closure**: if a kept token is built by merging `a + b`, both parts are kept too,
   recursively. By construction, a text whose original tokenization used only kept tokens is tokenized
   the same way by the trimmed tokenizer.
3. **Byte fallback stays intact**: the 256 byte-level base tokens are always kept, so removed languages
   can still be encoded and decoded, with more tokens.
4. **Row slicing, no retraining**: kept rows of every vocab-sized tensor are copied unchanged; transformer
   layers are untouched.
5. **Tokenization check**: held-out text is tokenized with both tokenizers and compared. Texts whose
   original tokenization contains a removed token are *skipped* (they are expected to differ) and
   reported through the token coverage figure. The run aborts before export if any compared text
   differs. This verifies the texts you give it, not all possible text.
6. **Optional logit check** (`--verify-logits` / `verify --logits`, needs the `verify` extra, loads both
   models in float32): reports the maximum absolute logit difference on kept tokens for a few held-out
   texts. `verify --logits` fails if it exceeds `--atol` (default 1e-3); `run --verify-logits` only reports
   the number. It is not part of the unit tests or CI.

Weight slicing works directly on `safetensors` bytes (memory-mapped, shard by shard, any dtype with a
known element size), so trimming itself needs neither a GPU nor torch.

## Results

Measured with `lingotrim run` on English + Polish + code (FineWeb-Edu, FineWeb-2, CodeParrot samples),
verified on 9,341 held-out text chunks (at least 500 characters each); no compared text differed in
either run.

| Model | Vocabulary | Params removed | Weights | Token coverage of held-out text |
|---|---|---|---|---|
| Qwen3.8-27B | 248,320 -> 135,657 (-45.4%) | 1,153,669,120 (~1.15B) | 55.56 GB -> 53.26 GB (-4.2%) | 99.915% |
| Qwen3-0.6B | 151,936 -> 100,430 (-33.9%) | 105,484,288 (~105M) | 1.50 GB -> 1.29 GB (-14.0%) | 99.923% |

Source: the `run-en-pl-code.json` summaries written by `lingotrim run` (weights are safetensors tensor
bytes, GB = 10^9 bytes). Token coverage is the share of tokens in the held-out text that are still in the
trimmed vocabulary. Texts containing the remaining tokens (under 0.1%; CJK and Hebrew tokens are among
the most frequent) were skipped by the tokenization check and would be encoded with byte-level fallback
tokens by the trimmed model.

Vocabulary trimming pays off most for small models and embedding/LM-head-bound workloads. For a 27B
model the embeddings are a few percent of the weights. It does not remove other languages' knowledge from
the transformer layers.

## Install

`lingotrim` is not published on PyPI. Install from GitHub with uv, or from a source checkout with pip:

```bash
uv tool install "lingotrim[corpus,gguf] @ git+https://github.com/grodzickir/lingotrim"
# or run without installing:
uvx --from "lingotrim[corpus,gguf] @ git+https://github.com/grodzickir/lingotrim" lingotrim --help

# from source
git clone https://github.com/grodzickir/lingotrim && cd lingotrim
pip install ".[corpus,gguf]"
```

Python 3.10 or newer. Extras: `corpus` (Hugging Face `datasets`, for language samples), `gguf` (torch +
transformers for llama.cpp's `convert_hf_to_gguf.py`), `verify` (torch + transformers for the logit
check). The core trim/verify path needs only `numpy`, `tokenizers`, `regex` and `huggingface_hub`.

### llama.cpp and Ollama

GGUF export, quantization and `--imatrix` use llama.cpp (`convert_hf_to_gguf.py`, `llama-quantize`,
`llama-imatrix`). Where lingotrim looks:

1. `--llama-cpp DIR` or the `LLAMA_CPP_DIR` environment variable: a llama.cpp checkout (needed for
   the converter script) and/or build (binaries in `DIR`, `DIR/bin` or `DIR/build/bin`);
2. otherwise it downloads the latest llama.cpp release source plus the prebuilt CPU binaries for your
   OS into `~/.cache/lingotrim` (`%LOCALAPPDATA%\lingotrim` on Windows; override with `LINGOTRIM_HOME`);
3. binaries not found in the llama.cpp directory are looked up on `PATH`.

A missing binary produces an error that says what to set. `--ollama NAME` needs the `ollama` executable on
`PATH`.

## Usage

```bash
# Everything in one go (rerun to resume: finished steps are skipped)
lingotrim run Qwen/Qwen3.8-27B --languages en,pl --out ./qwen3.8-enpl \
    --quantize Q4_K_M Q5_K_M --ollama qwen3.8-27b-enpl

# Local model dir, your own corpus source, no GGUF
lingotrim run ./my-model -l en,de --source legal=my-org/legal-corpus@body --out ./out

# Individual steps
lingotrim inspect ./model                                 # vocabulary by Unicode script
lingotrim corpus en pl code --out corpus --chars 20000000 # writes corpus/fit and corpus/heldout
lingotrim trim ./model ./model-enpl --corpus corpus/fit
lingotrim verify ./model ./model-enpl --corpus corpus/heldout [--logits]
lingotrim export ./model ./model-enpl --out model-enpl.gguf --quantize Q4_K_M --ollama my-model
```

`run` writes `run-<languages>.json` (summary) and `lingotrim_report.json` (why each token was kept,
sliced tensors, bytes saved) into the output directory.

### Importance-matrix calibration (`--imatrix`)

Quantization can be guided by an importance matrix computed from calibration text with llama.cpp's
`llama-imatrix`, which is passed to `llama-quantize --imatrix`. It requires `llama-imatrix` (shipped in
llama.cpp releases and builds) and only has an effect together with `--quantize`.

```bash
# run: calibrates on text taken from the fit corpus (up to 4M characters, split across sources)
lingotrim run Qwen/Qwen3-0.6B -l en,pl --quantize Q4_K_M --imatrix

# export: calibrate on your own text file
lingotrim export ./model ./model-enpl --out model-enpl.gguf --quantize Q4_K_M --imatrix calib.txt

# standalone
lingotrim imatrix model-enpl-bf16.gguf --corpus calib.txt --out model.imatrix [--ctx-size 512]
```

No claim is made here about how much `--imatrix` improves quality; evaluate it on your own tasks.

### Languages

`en pl de fr es it pt nl sv da no fi cs sk hu ro hr sl lt lv et tr vi id ms ca uk ru bg sr el he ar fa
hi bn ta th zh ja ko`, plus any raw FineWeb-2 config such as `pol_Latn`, and custom sources via
`--source name=dataset[:config][@column]`.

## Supported models and limitations

- **Tokenizer**: byte-level BPE in Hugging Face `tokenizer.json` format with `safetensors` weights.
  Other tokenizers (e.g. SentencePiece) are rejected with an error.
- **Tested only on Qwen-family models** (Qwen3-0.6B and Qwen3.8-27B end to end, plus a small synthetic
  model in the unit tests). Other byte-level BPE models may work but have not been tried.
- Token ids in `config.json`, `generation_config.json` (including nested `text_config`) and
  `tokenizer_config.json` are remapped. GGUF export pins the original pre-tokenizer type, which
  llama.cpp's converter would otherwise fail to detect for a trimmed vocabulary.
- **Verification scope**: only the held-out text you supply is compared, and texts containing removed
  tokens are skipped (see token coverage above). The optional logit check covers a handful of texts on
  small models and is not exercised by CI.
- Not supported: block-quantized checkpoints (scale tensors are not sliced), vision projectors
  (`--mmproj`; only the text model is exported).
- **Hardware**: lingotrim does not use a GPU. Trimming streams safetensors shards. GGUF conversion runs
  llama.cpp's converter on CPU through torch and needs enough RAM for the model; RAM needs were not
  measured here.
- Downloading models, corpora and llama.cpp needs network access.

## Platforms

Windows 11 is where the tool was developed and used for the runs above. The code avoids OS-specific
assumptions (paths via `pathlib`, UTF-8/LF files, no shell invocation) and CI runs ruff and the unit tests
on Ubuntu, macOS and Windows with Python 3.10 and 3.12. The CI tests mock llama.cpp and Ollama: the full
pipeline including the llama.cpp download and conversion has not been run on Linux or macOS.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
```

## License

MIT, see [LICENSE](LICENSE).
