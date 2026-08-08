#!/usr/bin/env node
// Hugging Face docs static site — EN + zh-CN, package tracks
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const EN_PAGES = path.join(ROOT, "docs", "pages");
const ZH_PAGES = path.join(ROOT, "docs", "zh", "pages");
const DIST = path.join(ROOT, "dist");
const BASE = (process.env.PAGES_BASE || "").replace(/\/$/, "");
const UI = JSON.parse(fs.readFileSync(path.join(__dirname, "i18n", "ui.json"), "utf8"));

const CHEV_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"></polyline></svg>';

const PACKAGE_ORDER = [
  "hub",
  "huggingface_hub",
  "cli",
  "transformers",
  "datasets",
  "diffusers",
  "tokenizers",
  "transformers.js",
  "peft",
  "accelerate",
  "trl",
  "optimum",
  "bitsandbytes",
  "safetensors",
  "timm",
  "smolagents",
  "inference-providers",
  "inference-endpoints",
  "text-generation-inference",
  "text-embeddings-inference",
  "dataset-viewer",
  "autotrain",
  "lerobot",
  "kernels",
  "xet",
];

const PACKAGE_LABEL = {
  hub: "Hub",
  huggingface_hub: "Hub Client",
  cli: "CLI",
  transformers: "Transformers",
  datasets: "Datasets",
  diffusers: "Diffusers",
  tokenizers: "Tokenizers",
  "transformers.js": "Transformers.js",
  peft: "PEFT",
  accelerate: "Accelerate",
  trl: "TRL",
  optimum: "Optimum",
  bitsandbytes: "BitsAndBytes",
  safetensors: "Safetensors",
  timm: "timm",
  smolagents: "Smolagents",
  "inference-providers": "Inference Providers",
  "inference-endpoints": "Inference Endpoints",
  "text-generation-inference": "TGI",
  "text-embeddings-inference": "TEI",
  "dataset-viewer": "Dataset Viewer",
  autotrain: "AutoTrain",
  lerobot: "LeRobot",
  kernels: "Kernels",
  xet: "Xet",
};

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function asset(p, locale = "en") {
  const rel = String(p).replace(/^\//, "");
  const isShared = rel.startsWith("assets/") || rel.startsWith("meta/");
  const locPrefix = !isShared && locale === "zh" ? "zh/" : "";
  const full = locPrefix + rel;
  return BASE ? `${BASE}/${full}` : `/${full}`;
}

function htmlEscape(s) {
  return String(s)
    .replace(/&/g, "&" + "amp;")
    .replace(/</g, "&" + "lt;")
    .replace(/>/g, "&" + "gt;")
    .replace(/"/g, "&" + "quot;");
}

function isHtmlDoc(text) {
  const t = String(text).trimStart().slice(0, 200).toLowerCase();
  return t.startsWith("<!doctype") || t.startsWith("<html") || t.startsWith("<head");
}

function walk(dir, acc = []) {
  if (!fs.existsSync(dir)) return acc;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, acc);
    else if (ent.isFile() && ent.name.endsWith(".md")) acc.push(p);
  }
  return acc;
}

function titleFromMd(md, fallback) {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].replace(/[`*]/g, "").trim() : fallback;
}

function humanize(slug) {
  return slug
    .replace(/\.md$/, "")
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function relToHtml(rel) {
  return rel.replace(/\.md$/, ".html");
}

function pageHref(rel, locale) {
  return asset(relToHtml(rel), locale);
}

function packageOf(rel) {
  if (rel === "index.md") return "home";
  const top = rel.split("/")[0];
  return top || "other";
}

function groupNameFor(rel, pkg) {
  if (pkg === "home") return "Home";
  const parts = rel.replace(/\.md$/, "").split("/");
  // package/foo/bar -> group by second segment
  if (parts.length <= 2) return "Guides";
  const seg = parts[1];
  // common HF patterns
  if (/^(en|main|v\d)/i.test(seg) && parts.length >= 3) {
    return parts[2] ? humanize(parts[2]) : "Guides";
  }
  if (/model_doc|models|api|reference/i.test(seg)) return "Reference";
  if (/tutorials?|training|generation|tasks/i.test(seg)) return "Tutorials";
  if (/pipeline|quickstart|installation|index/i.test(parts[parts.length - 1])) return "Getting started";
  return humanize(seg);
}

function buildNav(files, locale) {
  const byTrack = new Map();
  byTrack.set("home", {
    id: "home",
    name: "Home",
    badge: "·",
    groups: [],
  });

  for (const pkg of PACKAGE_ORDER) {
    byTrack.set(pkg, {
      id: pkg,
      name: PACKAGE_LABEL[pkg] || pkg,
      badge: "▸",
      groups: [],
    });
  }
  byTrack.set("other", { id: "other", name: "Other", badge: "·", groups: [] });

  for (const f of files) {
    const pkg = packageOf(f.rel);
    const track = byTrack.get(pkg) || byTrack.get("other");
    const gName = groupNameFor(f.rel, pkg);
    let g = track.groups.find((x) => x.name === gName);
    if (!g) {
      g = { name: gName, items: [] };
      track.groups.push(g);
    }
    g.items.push({
      title: f.title,
      href: pageHref(f.rel, locale),
      rel: f.rel,
    });
  }

  const tracks = [];
  for (const id of ["home", ...PACKAGE_ORDER, "other"]) {
    const t = byTrack.get(id);
    if (!t) continue;
    const itemCount = t.groups.reduce((n, g) => n + g.items.length, 0);
    if (itemCount === 0) continue;
    // cap huge groups: sort items
    for (const g of t.groups) {
      g.items.sort((a, b) => a.title.localeCompare(b.title));
    }
    t.groups.sort((a, b) => a.name.localeCompare(b.name));
    tracks.push({
      id: t.id,
      name: t.name,
      badge: t.badge,
      groups: t.groups,
      count: itemCount,
    });
  }
  return tracks;
}

function renderNavHtml(tracks, activeRel) {
  // Ultra-slim: only track headers. Items hydrate from assets/nav.json on expand.
  const parts = [];
  const activePkg = activeRel ? packageOf(activeRel) : "home";
  for (const track of tracks) {
    const trackActive =
      track.id === activePkg ||
      track.groups.some((g) => g.items.some((it) => it.rel === activeRel));
    const open = trackActive || track.id === "home" || track.id === "hub" ? "1" : "0";
    parts.push(
      `<div class="track" data-track="${htmlEscape(track.id)}" data-open="${open}" data-active="${trackActive ? "1" : "0"}" data-hydrated="0">`,
    );
    parts.push(
      `<button type="button" class="track-btn" data-track-toggle="${htmlEscape(track.id)}" aria-expanded="${open === "1"}" data-needs-items="1"><span class="chev">${CHEV_SVG}</span><span class="track-label">${htmlEscape(track.name)}</span><span class="track-count">${track.count}</span></button>`,
    );
    parts.push(
      `<div class="track-panel"><div class="track-panel-inner"><div class="track-body"><div class="muted" style="padding:0.45rem 0.5rem;font-size:0.78rem">Loading…</div></div></div></div></div>`,
    );
  }
  return parts.join("\n");
}

function enhanceCode(html) {
  return html
    .replace(
      /<pre><code class="language-([^"]*)">([\s\S]*?)<\/code><\/pre>/g,
      (_, lang, code) =>
        `<div class="code-block"><div class="code-bar"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="lang">${htmlEscape(lang || "text")}</span><button type="button" class="copy-btn" data-copy>Copy</button></div><pre><code class="language-${htmlEscape(lang)}">${code}</code></pre></div>`,
    )
    .replace(
      /<pre><code>([\s\S]*?)<\/code><\/pre>/g,
      (_, code) =>
        `<div class="code-block"><div class="code-bar"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="lang">text</span><button type="button" class="copy-btn" data-copy>Copy</button></div><pre><code>${code}</code></pre></div>`,
    );
}

function tocFromHtml(html) {
  const items = [];
  const re = /<h([23])\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g;
  let m;
  while ((m = re.exec(html))) {
    const text = m[3].replace(/<[^>]+>/g, "").trim();
    if (text) items.push({ level: Number(m[1]), id: m[2], text });
  }
  if (items.length < 2) return "";
  return `<nav class="toc"><div class="toc-title">On this page</div><ul>${items
    .map(
      (it) =>
        `<li class="l${it.level}"><a href="#${htmlEscape(it.id)}">${htmlEscape(it.text)}</a></li>`,
    )
    .join("")}</ul></nav>`;
}

function postProcessHtml(html, fromRel, locale) {
  return html.replace(/href="([^"]+)"/g, (full, href) => {
    if (!href || href.startsWith("#") || href.startsWith("mailto:") || href.startsWith("data:")) return full;
    // external non-docs
    if (/^https?:\/\//i.test(href) && !href.includes("huggingface.co/docs/")) return full;
    // HF docs absolute -> local
    let target = href;
    let hash = "";
    const hi = target.indexOf("#");
    if (hi >= 0) {
      hash = target.slice(hi);
      target = target.slice(0, hi);
    }
    if (/^https?:\/\/huggingface\.co\/docs\//i.test(target)) {
      let rel = target.replace(/^https?:\/\/huggingface\.co\/docs\//i, "");
      if (rel.endsWith(".md")) rel = rel.slice(0, -3);
      if (!rel.endsWith(".html")) rel = rel + ".html";
      return `href="${asset(rel, locale)}${hash}"`;
    }
    // relative .md
    if (target.endsWith(".md") || (!target.includes(".") && !target.startsWith("/"))) {
      const dir = path.posix.dirname(fromRel.replace(/\\/g, "/"));
      let rel = target.replace(/^\.\//, "");
      if (!rel.startsWith("/")) {
        rel = path.posix.normalize(path.posix.join(dir === "." ? "" : dir, rel));
      }
      rel = rel.replace(/^\/+/, "");
      if (rel.endsWith(".md")) rel = rel.slice(0, -3) + ".html";
      else if (!rel.endsWith(".html")) rel = rel + ".html";
      return `href="${asset(rel, locale)}${hash}"`;
    }
    return full;
  });
}

function layout({ locale, title, bodyHtml, navHtml, tocHtml, rel, ui, mtBanner }) {
  const enHref = asset(relToHtml(rel), "en");
  const zhHref = asset(relToHtml(rel), "zh");
  const activeEn = locale === "en" ? " active" : "";
  const activeZh = locale === "zh" ? " active" : "";
  const langAttr = locale === "zh" ? "zh-CN" : "en";

  return `<!DOCTYPE html>
<html lang="${langAttr}" data-locale="${locale}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${htmlEscape(title)} · ${htmlEscape(ui.brand)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css" />
  <link rel="stylesheet" href="${asset("assets/site.css")}" />
  <link rel="alternate" hreflang="en" href="${enHref}" />
  <link rel="alternate" hreflang="zh-CN" href="${zhHref}" />
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <button type="button" class="menu-btn" id="menuBtn" aria-label="${htmlEscape(ui.menu)}">${htmlEscape(ui.menu)}</button>
      <a class="brand" href="${asset("index.html", locale)}">
        <span class="brand-mark">🤗</span>
        <span class="brand-text">${htmlEscape(ui.brand)}</span>
        <span class="brand-v">${htmlEscape(ui.brandSub)}</span>
      </a>
      <nav class="chips" id="trackChips" aria-label="Tracks"></nav>
      <div class="lang-switch" role="navigation" aria-label="Language">
        <a class="lang-btn${activeEn}" href="${enHref}" data-lang-set="en" hreflang="en">${htmlEscape(ui.langEn)}</a>
        <a class="lang-btn${activeZh}" href="${zhHref}" data-lang-set="zh" hreflang="zh-CN">${htmlEscape(ui.langZh)}</a>
      </div>
      <a class="top-link" href="https://huggingface.co/docs" rel="noopener" target="_blank">${htmlEscape(ui.official)}</a>
    </div>
  </header>
  <div class="shell">
    <aside class="sidebar" id="sidebar">
      <div class="side-head">
        <div class="search-wrap">
          <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3-3"/></svg>
          <input class="search" id="search" type="search" placeholder="${htmlEscape(ui.searchPlaceholder)}" autocomplete="off" />
        </div>
        <div class="side-label">${htmlEscape(ui.learningPath)}</div>
      </div>
      <nav class="nav" id="nav">${navHtml}</nav>
      <div class="side-foot">${htmlEscape(ui.footer)}</div>
    </aside>
    <button type="button" class="backdrop" id="backdrop" aria-label="Close menu"></button>
    <main class="main">
      <div class="content-wrap">
        <article class="content prose">
          ${mtBanner || ""}
          ${bodyHtml}
          <p class="page-foot">${htmlEscape(ui.footer)}</p>
        </article>
        ${tocHtml}
      </div>
    </main>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
  <script src="${asset("assets/site.js")}"></script>
  <script>window.__HF_ACTIVE_REL__=${JSON.stringify(rel)};</script>
  <script>document.querySelectorAll("pre code").forEach((el)=>window.hljs&&hljs.highlightElement(el));</script>
</body>
</html>`;
}

function copyAssets() {
  const out = path.join(DIST, "assets");
  ensureDir(out);
  for (const f of ["site.css", "site.js"]) {
    fs.copyFileSync(path.join(__dirname, "site-assets", f), path.join(out, f));
  }
  fs.copyFileSync(path.join(__dirname, "i18n", "ui.json"), path.join(out, "ui.json"));
  ensureDir(path.join(DIST, "meta"));
  if (fs.existsSync(path.join(ROOT, "docs", "llms.txt"))) {
    fs.copyFileSync(path.join(ROOT, "docs", "llms.txt"), path.join(DIST, "meta", "llms.txt"));
  }
  if (fs.existsSync(path.join(ROOT, "docs", "list.json"))) {
    fs.copyFileSync(path.join(ROOT, "docs", "list.json"), path.join(DIST, "meta", "list.json"));
  }
  // package llms indexes
  const llmsDir = path.join(ROOT, "docs", "llms");
  if (fs.existsSync(llmsDir)) {
    const metaLlms = path.join(DIST, "meta", "llms");
    ensureDir(metaLlms);
    for (const f of fs.readdirSync(llmsDir)) {
      fs.copyFileSync(path.join(llmsDir, f), path.join(metaLlms, f));
    }
  }
}

function loadPages(rootDir) {
  const files = walk(rootDir);
  const pages = [];
  for (const abs of files) {
    const rel = path.relative(rootDir, abs).replace(/\\/g, "/");
    if (rel.endsWith("/_full.md") || rel === "_full.md") continue;
    if (fs.statSync(abs).size > 1500000) continue;
    let md = fs.readFileSync(abs, "utf8");
    if (isHtmlDoc(md)) continue;
    md = md.replace(/^<!-- huggingface-docs:[\s\S]*?-->\n*/m, "");
    md = md.replace(/^<!-- kaggle-docs:[\s\S]*?-->\n*/m, "");
    const title = titleFromMd(md, humanize(path.basename(rel, ".md")));
    pages.push({ abs, rel, md, title });
  }
  return pages;
}

function buildLocale(locale, pages, navTracks) {
  const ui = UI[locale] || UI.en;
  const outRoot = locale === "zh" ? path.join(DIST, "zh") : DIST;
  ensureDir(outRoot);
  let n = 0;
  for (const page of pages) {
    const nav = renderNavHtml(navTracks, page.rel);
    marked.setOptions({ gfm: true, breaks: false });
    let body = marked.parse(page.md);
    body = enhanceCode(body);
    body = postProcessHtml(body, page.rel, locale);
    const toc = tocFromHtml(body);
    const mtBanner =
      locale === "zh"
        ? `<div class="mt-banner">${htmlEscape(ui.mtBanner)} <a href="${asset(relToHtml(page.rel), "en")}">${htmlEscape(ui.mtViewEn)}</a></div>`
        : "";
    const html = layout({
      locale,
      title: page.title,
      bodyHtml: body,
      navHtml: nav,
      tocHtml: toc,
      rel: page.rel,
      ui,
      mtBanner,
    });
    const outFile = path.join(outRoot, relToHtml(page.rel));
    ensureDir(path.dirname(outFile));
    fs.writeFileSync(outFile, html);
    n++;
  }
  return n;
}

function main() {
  fs.rmSync(DIST, { recursive: true, force: true });
  ensureDir(DIST);
  copyAssets();

  const enPages = loadPages(EN_PAGES);
  if (!enPages.length) {
    console.error("No EN pages found under docs/pages");
    process.exit(1);
  }

  const zhPages = enPages.map((p) => {
    const zhAbs = path.join(ZH_PAGES, p.rel);
    if (fs.existsSync(zhAbs)) {
      let md = fs.readFileSync(zhAbs, "utf8");
      md = md.replace(/^<!--[\s\S]*?-->\n*/m, "");
      if (!isHtmlDoc(md) && md.trim().length > 20) {
        return { ...p, md, title: titleFromMd(md, p.title) };
      }
    }
    return { ...p };
  });

  const enNav = buildNav(enPages, "en");
  const zhNav = buildNav(zhPages, "zh");
  fs.writeFileSync(path.join(DIST, "assets", "nav.json"), JSON.stringify(enNav, null, 2));
  fs.writeFileSync(path.join(DIST, "assets", "nav.zh.json"), JSON.stringify(zhNav, null, 2));

  const nEn = buildLocale("en", enPages, enNav);
  const nZh = buildLocale("zh", zhPages, zhNav);

  const counts = {};
  for (const t of enNav) counts[t.name] = t.count;
  console.log(
    `[en] ${nEn} pages — ${Object.entries(counts)
      .map(([k, v]) => `${k}:${v}`)
      .join(" · ")}`,
  );
  console.log(`[zh] ${nZh} pages (EN fallback where untranslated)`);
  console.log(`Built locales en+zh -> ${DIST} (BASE=${BASE || "/"})`);
}

main();
