# 03 — Technical Architecture

## The hard problem

"Interactive visualisation of real ML" normally fails one of three ways:
1. **Fake it** — hand-animated SVGs that drift from reality and teach wrong intuitions.
2. **Ship a server** — every interaction hits a GPU backend; it costs money, it dies under
   traffic, and nobody can self-host it.
3. **Ship a notebook** — technically real, pedagogically inert, and 90% of visitors bounce.

Our answer is a **three-tier execution model**. Each tier has a clear rule for when it applies,
and a mandatory fallback to the tier above it.

### Tier 1 — Precomputed traces (default, always available)
Real PyTorch runs offline, captured with hooks, serialised to a compact binary format, served
from a CDN. The browser scrubs through *actual recorded numbers* — attention maps, residual
norms, gradients, routing decisions, loss curves, profiler timelines.

- Cost: zero at runtime. Works offline, works on a phone, works with JS-lite.
- Limit: you can only explore inputs that were captured. We mitigate with wide parameter
  sweeps (e.g. 200 prompts × 12 layers × 12 heads for GPT-2) and smart interpolation.
- **Every trace is reproducible**: the capture script lives next to it, CI re-runs a sample
  and checks it still matches.

### Tier 2 — Live in-browser compute (interactive, no backend)
Real models and real kernels executing in the visitor's browser.

- **WebGPU** for small models (GPT-2 124M, ViT-Tiny, tiny diffusion UNet, MNIST-scale training)
  via ONNX Runtime Web / transformers.js, plus our own WGSL compute shaders for teaching
  kernels (tiled matmul, online softmax, scan).
- **Rust → WASM** for deterministic simulators: GPU execution simulator, roofline engine,
  serving discrete-event simulator, HNSW builder, autodiff tracer. Deterministic across
  machines, fast, and testable against a Python oracle.
- **Pyodide** for "edit this Python and run it" cells on numpy-scale examples.
- Limit: no CUDA, no profiling, no large models. Falls back to Tier 1 when WebGPU is absent.

### Tier 3 — Real GPU sandbox (opt-in, rate-limited, optional to self-host)
For things that only mean something on real silicon: Triton/CUDA compilation, Nsight-style
profiling, the matmul ladder's measured timings, quantisation accuracy runs.

- Short-lived container, one GPU, hard 30 s wall clock, no network egress, seccomp-confined,
  read-only base image, output size capped.
- Aggressive content-addressed caching: identical kernel source → cached result. Most visitors
  hit cache, so marginal cost tends to zero.
- Anonymous users get a small daily quota; signed-in users more; **BYO-GPU** via a self-hosted
  runner token for power users; the whole runner is optional — the site degrades to showing
  pre-measured reference results from CI.

**Rule:** no concept may *require* Tier 3 to be understood. Tier 3 adds proof, never meaning.

---

## Stack decisions

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Repo | pnpm workspaces + Turborepo | Many packages, heavy caching, cheap CI |
| Site | **Astro 5** + React islands | Content-heavy site that must be fast on a weak connection; islands mean a page with one lab ships one lab's JS, not the whole app. Next.js would hydrate far more for no gain here |
| Content | **MDX** + Zod-validated frontmatter | Authors write prose with embedded labs; the schema is enforced at build, so broken content cannot merge |
| Labs | React 19 + TypeScript + Zustand | Labs are genuinely stateful apps; signals/solid would fragment the contributor pool |
| Charts | **visx** (+ D3 scales) | Composable primitives; we need custom marks constantly, so a chart *library* (Recharts/Chart.js) would fight us |
| Canvas/3D | WebGL2 via regl for 2D-heavy, **Three.js** for 3D | Tensor heatmaps at 10⁵ cells need GPU drawing, not SVG |
| Numerics (browser) | WebGPU compute + custom WGSL; ONNX Runtime Web for models | Only path to real model execution client-side |
| Simulators | **Rust → wasm-bindgen** | Determinism, speed, and the same crate is testable in CI against Python references |
| Python side | `mechanist-trace` (PyTorch hooks → traces), `mechanist-ref` (reference impls) | Traces and golden tests must come from the real framework |
| Trace format | **`.mtr`** = zstd-compressed safetensors payload + JSON manifest, HTTP range-readable | Streams layer-by-layer; a 12-layer GPT-2 attention trace is ~3 MB, not 300 |
| Storage | Cloudflare R2 (traces, large assets) | No egress fees; this matters when shipping tensors |
| Hosting | Cloudflare Pages (static) + Workers (thin API) | Global, cheap, no server to babysit |
| Runner | FastAPI + container pool on Modal / RunPod, pluggable | Serverless GPUs mean we pay per second, not per hour |
| Search | **Orama** (client-side index) + semantic search over concept embeddings | No search server; works offline |
| Graph | Concept graph built at compile time → JSON + SQLite (WASM) for queries | Instant graph traversal client-side |
| Auth/progress | Optional. Local-first (IndexedDB), optional sync via a Worker + Turso | Progress must work with zero account |
| Testing | Vitest, Playwright, pytest, **numerical golden tests** | See "Correctness system" below |
| Docs/i18n | Astro i18n, content keyed by locale | Translation is a first-class contribution type |

### Repository layout

```
mechanist/
├── apps/
│   ├── web/                  # Astro site: content, routing, graph views, search
│   └── runner/               # Optional GPU sandbox service (FastAPI + container pool)
├── packages/
│   ├── lab-kit/              # React SDK for labs (the contributor's main surface)
│   ├── viz/                  # visx/WebGL primitives: TensorView, Heatmap, Roofline, GraphView
│   ├── engine-cost/          # FLOPs / bytes / memory / latency cost model (TS, shared)
│   ├── engine-sim/           # Rust→WASM: GPU sim, serving sim, scan, HNSW, autodiff tape
│   ├── webgpu-kernels/       # Teaching kernels in WGSL + their JS harness
│   ├── trace-client/         # .mtr reader, streaming, caching
│   ├── content-schema/       # Zod schemas + graph builder + link checker
│   └── ui/                   # Design system, theming, a11y primitives
├── python/
│   ├── mechanist_trace/      # Hook-based capture → .mtr
│   ├── mechanist_ref/        # <300-line reference impls of every concept
│   └── mechanist_bench/      # Measured benchmarks (matmul ladder, kernels) → CI artefacts
├── content/
│   ├── concepts/<slug>/      # The unit of contribution (see 04)
│   ├── papers/               # One YAML per paper
│   ├── paths/                # Curated learning paths
│   └── glossary/
├── traces/                   # Capture scripts; artefacts live in R2
└── docs/                     # This plan, contributor guides, ADRs
```

---

## The cost engine (`engine-cost`)

A single shared model used by the roofline lab, memory calculator, scaling-law calculator,
serving simulator, and every "Systems" lens on every page. One source of truth means a number
cannot be right on one page and wrong on another.

```ts
interface ModelSpec {
  nLayers: number; dModel: number; nHeads: number; nKVHeads: number;
  dHead: number; dFFN: number; vocabSize: number; seqLen: number;
  attention: 'mha' | 'mqa' | 'gqa' | 'mla' | 'sliding';
  ffn: 'mlp' | 'swiglu' | 'moe';
  moe?: { nExperts: number; topK: number; sharedExperts: number };
  dtype: DType; kvDtype: DType;
}

interface HardwareSpec {       // sourced, dated, citation-linked
  name: string; flops: Record<DType, number>;   // dense & sparse
  hbmBandwidth: number; hbmCapacity: number;
  l2Size: number; smemPerSM: number; numSMs: number;
  interconnect: { intraNode: number; interNode: number };
}

analyze(model, hardware, workload) -> {
  flops: PerOpBreakdown;         // fwd, bwd, by op
  bytes: PerOpBreakdown;         // HBM traffic
  arithmeticIntensity: number;   // per op and aggregate
  bound: 'compute' | 'memory' | 'latency';
  memory: { params, grads, optimizer, activations, kvCache, workspace };
  time: { estimateMs, mfu, confidence };
  commVolume?: PerStrategy;      // for a given parallelism plan
}
```

Every hardware entry carries a `source` URL and `measuredBy` field. Estimates are labelled
**estimate**; measured numbers are labelled **measured on \<GPU\>, \<date\>, \<commit\>**. We never
blur the two — that distinction is itself a lesson.

---

## Correctness system (the thing that makes this trustworthy)

An open-source learning platform lives or dies on whether people believe its numbers.

1. **Golden numerical tests.** Every browser kernel (WGSL, WASM) and every reference impl is
   tested against PyTorch within a stated tolerance, in CI. `webgpu-kernels/flash_attention`
   must match `torch.nn.functional.scaled_dot_product_attention` to 1e-3 or the build fails.
2. **Trace reproducibility.** Each `.mtr` records model revision, seed, library versions and
   a capture script hash. A nightly job re-captures a sample and diffs.
3. **Claim provenance in frontmatter.** Numerical claims in prose carry a `verified` block:
   ```yaml
   claims:
     - text: "GQA with 8 KV heads cuts KV cache by 4x vs MHA at 32 heads"
       kind: derived          # derived | measured | cited
       check: tests/gqa_kv_ratio.test.ts
   ```
   `kind: cited` requires a DOI/arXiv link; `kind: measured` requires a benchmark artefact.
4. **Page badge.** Readers see `✓ numerics verified` / `⚠ estimate` / `↗ cited` per claim.
   A page with unverified numerics cannot be marked `stable`.
5. **Link & graph integrity.** Build fails on a dangling prerequisite, orphan concept, cycle in
   the prerequisite DAG, or a paper reference with no matching entry.
6. **Performance budgets.** A lab must reach interactive in < 2.5 s on a mid-tier laptop and
   ship < 250 KB JS beyond the shared runtime. CI measures it.
7. **Accessibility gates.** Every visualisation needs a non-visual equivalent (data table,
   text summary), keyboard operation, and colour choices that survive colour-blind simulation.
   Automated axe checks plus a manual rubric item.

---

## Performance & delivery

- Static HTML for all prose; labs hydrate on viewport entry (`client:visible`).
- Traces stream by layer with HTTP range requests; the UI renders progressively.
- WebGPU detected at runtime; graceful degradation path is explicit per lab, never a blank box.
- Full offline mode: service worker caches visited concepts and their Tier-1 traces; a CLI
  (`npx mechanist offline <path>`) pre-bundles an entire learning path for airplane/low-connectivity use.
- Self-host: `docker compose up` gives the full site minus Tier 3; one env var points at your
  own GPU runner.

## Security model

- Runner: no network egress, non-root, read-only rootfs, seccomp + gVisor, 30 s CPU/wall limit,
  memory cap, output cap, per-IP and per-account quotas, queue backpressure.
- Submitted kernel code is never executed outside the sandbox and never persisted beyond cache.
- Content is MDX compiled at build time only — no runtime MDX evaluation from user input.
- Pyodide and WebGPU run in the visitor's own browser sandbox; no user code reaches our servers
  except via the runner path.
