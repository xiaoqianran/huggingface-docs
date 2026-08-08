#!/usr/bin/env node
/**
 * Fetch Hugging Face docs:
 *  1) per-package llms.txt (TOC)
 *  2) per-package llms-full.txt (full Markdown corpus)
 *  3) split full dumps into pages via scripts/split-from-full.py
 *
 * Avoids per-page HTTP (rate limits). Writes docs/llms + docs/pages.
 */
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DOCS = path.join(ROOT, "docs");
const LLMS_DIR = path.join(DOCS, "llms");
const TIMEOUT_MS = Math.max(10000, Number(process.env.FETCH_TIMEOUT_MS || 120000));
const UA =
  process.env.FETCH_UA ||
  "huggingface-docs-mirror/1.0 (+https://github.com/xiaoqianran/huggingface-docs)";

const PACKAGES = (
  process.env.HF_PACKAGES ||
  [
    "hub","huggingface_hub","cli","transformers","datasets","diffusers","tokenizers",
    "transformers.js","peft","accelerate","trl","optimum","bitsandbytes","safetensors",
    "timm","smolagents","inference-providers","inference-endpoints",
    "text-generation-inference","text-embeddings-inference","dataset-viewer",
    "autotrain","lerobot","kernels","xet",
  ].join(",")
)
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function isHtml(text) {
  const t = String(text).trimStart().slice(0, 200).toLowerCase();
  return t.startsWith("<!doctype") || t.startsWith("<html") || t.startsWith("<head");
}

async function fetchText(url, { accept, attempt = 0 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      signal: ctrl.signal,
      headers: {
        "User-Agent": UA,
        Accept: accept || "text/plain, text/markdown;q=0.9, */*;q=0.1",
      },
      redirect: "follow",
    });
    if (res.status === 429 || res.status === 503) {
      const ra = Number(res.headers.get("retry-after") || 0);
      const backoff = Math.min(90000, (ra > 0 ? ra * 1000 : 3000 * Math.pow(2, attempt)) + Math.random() * 500);
      if (attempt < 10) {
        console.warn(`rate ${res.status} ${url} — sleep ${Math.round(backoff)}ms`);
        await sleep(backoff);
        return fetchText(url, { accept, attempt: attempt + 1 });
      }
      throw new Error(`HTTP ${res.status} for ${url}`);
    }
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const text = new TextDecoder("utf-8").decode(await res.arrayBuffer());
    return text;
  } finally {
    clearTimeout(timer);
  }
}

async function main() {
  ensureDir(LLMS_DIR);
  console.log("Packages:", PACKAGES.join(", "));

  let ok = 0;
  for (const pkg of PACKAGES) {
    const base = `https://huggingface.co/docs/${pkg}`;
    try {
      const index = await fetchText(`${base}/llms.txt`);
      if (isHtml(index) || index.trim().length < 20) throw new Error("bad llms.txt");
      fs.writeFileSync(path.join(LLMS_DIR, `${pkg}.txt`), index);
      await sleep(250);
      const full = await fetchText(`${base}/llms-full.txt`);
      if (isHtml(full) || full.trim().length < 40) throw new Error("bad llms-full.txt");
      fs.writeFileSync(path.join(LLMS_DIR, `${pkg}.full.txt`), full);
      ok++;
      console.log(`ok  ${pkg.padEnd(28)} index=${index.length} full=${full.length}`);
    } catch (e) {
      console.warn(`fail ${pkg}: ${e.message}`);
    }
    await sleep(400);
  }

  console.log(`\nFetched ${ok}/${PACKAGES.length} packages — splitting…`);
  const r = spawnSync("python3", [path.join(__dirname, "split-from-full.py")], {
    cwd: ROOT,
    stdio: "inherit",
  });
  if (r.status !== 0) process.exit(r.status || 1);
  if (ok < 5) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
