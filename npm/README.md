# lingotrim (npm wrapper)

```bash
npx lingotrim run Qwen/Qwen3-0.6B --languages en,pl --quantize Q4_K_M --ollama qwen3-0.6b-enpl
```

Thin launcher: it uses [`@manzt/uv`](https://github.com/manzt/uv-npm) to run the Python package
`lingotrim` in an isolated environment (Python is downloaded automatically if missing).

Set `LINGOTRIM_SPEC` to run a different build, e.g. a git checkout:

```bash
LINGOTRIM_SPEC="lingotrim[corpus,gguf] @ git+https://github.com/grodzickir/lingotrim" npx lingotrim --help
```

Full documentation: https://github.com/grodzickir/lingotrim
