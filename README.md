# Hugging Face Docs Mirror

Unofficial mirror of high-value **Hugging Face documentation**.

## Sources

Per-package [`llms.txt`](https://huggingface.co/docs/hub/llms.txt) indexes + Markdown pages from `huggingface.co/docs/*`.

Included packages: Hub, huggingface_hub, Transformers, Datasets, Diffusers, PEFT, Accelerate, TRL, Smolagents, Inference, TGI/TEI, and more.

## Local

```bash
npm install --no-save marked@15
npm run fetch
# optional (large): npm run translate
PAGES_BASE=/huggingface-docs npm run build
node scripts/serve-pages.mjs
```

## GitHub Actions

Daily fetch → optional zh cache → build → GitHub Pages.

## LLM / agent access ([llmstxt.org](https://llmstxt.org/))

| File | Purpose |
|------|---------|
| [`/llms.txt`](./llms.txt) | Curated page index (mirror URLs) |
| [`/llms-full.txt`](./llms-full.txt) | Full markdown corpus for ingestion |
| `/meta/llms-index.json` | Machine-readable page list |

Generated at build time from scraped pages.
