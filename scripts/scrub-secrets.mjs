#!/usr/bin/env node
/**
 * Scrub credential-like strings from docs trees so GitHub push protection
 * does not reject bot commits of mirrored documentation.
 *
 * Usage: node scripts/scrub-secrets.mjs [dir ...]
 * Default dirs: docs/pages docs/zh docs/llms
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

export function scrubSecrets(text) {
  let s = String(text ?? "");
  // Common provider tokens
  s = s.replace(/\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b/g, "$1_REDACTED");
  s = s.replace(/\bgithub_pat_[A-Za-z0-9_]{20,}\b/g, "github_pat_REDACTED");
  s = s.replace(/\bhf_[A-Za-z0-9]{16,}\b/g, "hf_REDACTED");
  s = s.replace(/\bsk-[A-Za-z0-9_\-]{16,}\b/g, "sk-REDACTED");
  s = s.replace(/\bxai-[A-Za-z0-9]{16,}\b/g, "xai-REDACTED");
  s = s.replace(/\bdckr_pat_[A-Za-z0-9_]+\b/g, "dckr_pat_REDACTED");
  s = s.replace(/\bAKIA[0-9A-Z]{16}\b/g, "AKIA_REDACTED");
  s = s.replace(/\bAIza[0-9A-Za-z\-_]{20,}\b/g, "AIza_REDACTED");
  s = s.replace(/\br8_[A-Za-z0-9]{20,}\b/g, "r8_REDACTED"); // Replicate
  s = s.replace(/\bsk-ant-[A-Za-z0-9_\-]{16,}\b/g, "sk-ant-REDACTED");
  s = s.replace(/\bsk-proj-[A-Za-z0-9_\-]{16,}\b/g, "sk-proj-REDACTED");
  // OpenAI-style project keys without sk- prefix are rare; keep assignment-based scrub
  s = s.replace(/\b(Bearer\s+)[A-Za-z0-9._\-]{20,}/gi, "$1REDACTED");
  // Assignment / env style: api_key="...."
  s = s.replace(
    /((?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret[_-]?key|mistral[_-]?api[_-]?key|openai[_-]?api[_-]?key|hf[_-]?token|token)\s*[=:]\s*["'`])([^"'`\s]{12,})(["'`])/gi,
    "$1REDACTED$3",
  );
  s = s.replace(
    /((?:api[_-]?key|apikey|access[_-]?token|secret|token)\s*[=:]\s*)([A-Za-z0-9_\-./+]{20,})/gi,
    "$1REDACTED",
  );
  // ENV exports
  s = s.replace(
    /\b([A-Z][A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|SECRET|TOKEN|PASSWORD))\s*=\s*["']?([^\s"']{12,})["']?/g,
    "$1=REDACTED",
  );
  // Mistral-ish long opaque keys in quotes near "mistral" context (best-effort)
  s = s.replace(
    /(mistral[^\n]{0,80})(["'`])([A-Za-z0-9_\-]{24,})(["'`])/gi,
    "$1$2REDACTED$4",
  );
  return s;
}

function walk(dir, acc = []) {
  if (!fs.existsSync(dir)) return acc;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, acc);
    else if (ent.isFile() && /\.(md|txt|json)$/i.test(ent.name)) acc.push(p);
  }
  return acc;
}

function scrubTree(dir) {
  let changed = 0;
  let scanned = 0;
  for (const abs of walk(dir)) {
    // drop full dumps entirely — they re-introduce secrets and bloat commits
    if (abs.endsWith(`${path.sep}_full.md`) || abs.endsWith("/_full.md")) {
      fs.unlinkSync(abs);
      changed++;
      console.log("rm  ", path.relative(ROOT, abs));
      continue;
    }
    scanned++;
    const raw = fs.readFileSync(abs, "utf8");
    const next = scrubSecrets(raw);
    if (next !== raw) {
      fs.writeFileSync(abs, next);
      changed++;
      console.log("scrub", path.relative(ROOT, abs));
    }
  }
  return { scanned, changed };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const args = process.argv.slice(2);
  const targets =
    args.length > 0
      ? args.map((a) => path.resolve(ROOT, a))
      : ["docs/pages", "docs/zh", "docs/llms"].map((d) => path.join(ROOT, d));

  let total = { scanned: 0, changed: 0 };
  for (const t of targets) {
    const r = scrubTree(t);
    total.scanned += r.scanned;
    total.changed += r.changed;
  }
  console.log(`[scrub] scanned=${total.scanned} changed=${total.changed}`);
}
