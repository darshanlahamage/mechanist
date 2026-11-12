# 04 — Content Schema & Lab Kit

The content model is the product. If adding a concept is hard, the community never forms; if
the schema is loose, quality rots. This document specifies both.

## The unit of contribution: a concept folder

```
content/concepts/flash-attention/
├── concept.yaml          # metadata, graph edges, claims, status
├── intuition.mdx         # depth 1 — 10 minutes, no math required
├── mechanism.mdx         # depth 2 — the default read, with the lab embedded
├── math.mdx              # depth 3 — derivation, every symbol shape-bound
├── systems.mdx           # cost model, roofline, scaling behaviour
├── failure.mdx           # how it breaks, reproducibly
├── lineage.mdx           # paper genealogy and what each one changed
├── code/
│   ├── from_scratch.py   # <300 lines, readable, tested
│   ├── production.py     # how the real library does it, annotated
│   └── kernel.triton.py  # the kernel, if applicable
├── labs/
│   └── tiler.tsx         # the interactive artefact
├── assessments.yaml      # predict / find-the-bug / diagnose items
└── tests/
    ├── test_from_scratch.py   # vs PyTorch reference, tolerance stated
    └── claims.test.ts         # verifies numerical claims in prose
```

Nothing else in the repo needs touching. The build discovers the folder, validates it,
wires it into the graph, generates routes, and indexes it for search.

## `concept.yaml`

```yaml
id: flash-attention
title: Flash Attention
tagline: Make attention IO-aware and the memory wall disappears.
track: accelerators
status: stable              # draft | review | stable | needs-update
difficulty: 4               # 1..5
estimatedMinutes: { intuition: 8, mechanism: 45, research: 90 }

prerequisites: [scaled-dot-product-attention, memory-hierarchy, online-softmax]
unlocks: [paged-attention, flash-attention-3, flex-attention]

edges:
  motivated-by:  [attention-quadratic-memory]     # the problem it solves
  alternative-to:[memory-efficient-attention, block-sparse-attention]
  superseded-by: []                               # honest about obsolescence
  used-in:       [vllm, pytorch-sdpa, llama-3-training]
  optimizes:     [scaled-dot-product-attention]

papers:
  - id: dao2022flashattention
    role: introduced
    oneIdea: Tile attention so the N×N matrix never materialises in HBM.
    delta: Prior work reduced FLOPs; this reduced HBM traffic, which was the real bound.
    heldUp: true
    caveat: Speedups are hardware- and shape-dependent; short sequences see little gain.

claims:
  - text: "Flash Attention reduces HBM accesses from O(N²) to O(N²d²/M)"
    kind: derived
    check: tests/claims.test.ts#io-complexity
  - text: "2.4x speedup over PyTorch SDPA at seq=4096, d=64 on A100"
    kind: measured
    artifact: bench/flash_a100_2024.json

labs:
  - id: tiler
    title: Flash Attention Tiler
    tier: 2                 # browser WebGPU
    fallback: trace         # if WebGPU unavailable
    flagship: true

maintainers: [github-handle]
reviewedBy: [github-handle]
lastVerified: 2026-11-01
```

### Status lifecycle

`draft` → `review` → `stable`. A concept reaches `stable` only when it has: all six lenses,
at least one lab, at least three assessment items, every claim verified, a11y pass, and two
reviewer approvals (one domain, one pedagogy). `needs-update` is set automatically when a
dependency changes or `lastVerified` is older than 12 months — **staleness is tracked, not hidden.**

## The six-lens template

Each lens file has a required shape so pages feel consistent and gaps are visible:

| File | Required sections |
|---|---|
| `intuition.mdx` | The problem in one paragraph · An analogy that is *honest about where it breaks* · One picture · "What you can now predict" |
| `mechanism.mdx` | Setup & shapes · Step-by-step with the lab inline · Worked numeric example · Common misconception, named and corrected |
| `math.mdx` | Notation table (symbol → shape → meaning) · Derivation with no skipped steps · Gradient · Edge cases · Equivalences to other formulations |
| `systems.mdx` | FLOPs · Bytes · Arithmetic intensity · Roofline position · Memory · Scaling behaviour · Measured numbers with hardware named |
| `failure.mdx` | Failure modes table (symptom → cause → fix) · At least one reproducible breakage · Diagnostic checklist |
| `lineage.mdx` | Timeline · What forced each change · What turned out wrong · Open questions |

The authoring CLI (`pnpm mechanist new concept`) scaffolds all of this pre-filled with prompts,
so the barrier is "fill in the blanks", not "invent a format".

## Lab Kit (`@mechanist/lab-kit`)

A lab should be ~100 lines of domain logic, not 1000 lines of plumbing. The kit supplies
everything that is not the idea itself.

### Primitives

| Component | Purpose |
|---|---|
| `<Lab>` | Shell: title, reset, share-state-as-URL, fullscreen, a11y announcer, perf budget guard |
| `<ParamPanel>` | Declarative controls (slider/select/toggle/vector/prompt) bound to typed state |
| `<Stepper>` | Timeline scrubbing over discrete algorithm steps, with play/pause and per-step captions |
| `<TensorView>` | Renders any tensor: heatmap, grid of numbers, histogram, or 3D; handles 10⁵+ cells on WebGL; shape badge; hover inspection; **always has a table fallback** |
| `<ShapeFlow>` | Shape algebra diagram: how dims transform through an op chain |
| `<CodeSync>` | Code block where the highlighted line follows the `<Stepper>` position |
| `<Roofline>` | Plots operating points against any hardware from `engine-cost` |
| `<CostReadout>` | Live FLOPs / bytes / memory / time for current parameters |
| `<GraphView>` | Concept or computation graph with focus + traversal |
| `<Trace>` | Loads a `.mtr`, exposes tensors as hooks, streams progressively |
| `<ModelRunner>` | Runs an ONNX/WebGPU model in-browser, exposes intermediates |
| `<KernelEditor>` | Triton/CUDA/WGSL editor → Tier-3 runner → result, diff vs reference, profile |
| `<SabotageToggle>` | Declares a "break it" variant and runs both side by side |
| `<Predict>` | Asks the learner to predict before revealing — the core pedagogical device |

### Example: a complete lab in the intended style

```tsx
export default function RopeLab() {
  const p = useParams({
    position: slider(0, 2048, 0),
    dim: select([64, 128, 256], 128),
    base: slider(1e3, 1e6, 1e4, { log: true }),
    scaling: select(['none', 'pi', 'ntk', 'yarn'], 'none'),
  })

  const { q, qRot, angles } = useCompute(() => ropeApply(p), [p])

  return (
    <Lab title="RoPE Visualiser" concept="rope">
      <ParamPanel params={p} />
      <Predict ask="If you double the position, what happens to the dot product with a key at position 0?" />
      <Split>
        <RotationPlanes vectors={q} rotated={qRot} angles={angles} />
        <TensorView tensor={qRot} label="q after rotation" />
      </Split>
      <Stepper steps={ropeSteps(p)}>
        <CodeSync src="code/from_scratch.py" />
      </Stepper>
      <CostReadout op="rope" params={p} />
    </Lab>
  )
}
```

Everything stateful is URL-encoded, so a learner can share *their exact configuration* —
including a broken one — in a GitHub issue or a lecture.

## Assessment schema

```yaml
- id: fa-predict-traffic
  kind: predict-number
  prompt: "seq=8192, d=64, M=100KB SRAM. How many times is K read from HBM?"
  answer: { value: 82, tolerance: 0.1, unit: "passes" }
  explain: "Q is tiled into ⌈N·d/M⌉ blocks; each block re-reads all of K."
  concept: flash-attention
  difficulty: 4

- id: fa-find-bug
  kind: find-the-bug
  code: labs/buggy/online_softmax_v3.py
  bugLine: 27
  symptom: "Correct for seq≤512, NaN above."
  explain: "Running max updated after the exponentiation, so exp overflows for large scores."

- id: moe-diagnose
  kind: diagnose
  given: { metrics: [router_entropy.json, expert_load.json, loss.json] }
  question: "Throughput halved at step 9k while loss kept falling. Why?"
  answer: expert-load-imbalance
  explain: "Entropy collapsed; one expert takes 60% of tokens, so one device serialises the batch."
```

Assessments feed a local-first spaced repetition scheduler keyed to *mechanisms*. Items are
contributed alongside content and reviewed with the same rigour.

## Paper entries

```yaml
id: dao2022flashattention
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
year: 2022
arxiv: 2205.14135
concepts: [flash-attention, online-softmax, memory-hierarchy]
oneIdea: Attention was memory-bound, not compute-bound; tile it so the score matrix stays in SRAM.
predecessors: [rabe2021memeff, milakov2018onlinesoftmax]
successors: [dao2023flashattention2, shah2024flashattention3]
whatHeldUp: "IO-awareness as the right lens for kernel design — now standard."
whatDidNot: "v1's backward pass partitioning was superseded within a year."
reproductionNotes: "Reported speedups assume A100 and head_dim ≤ 128; short sequences regress."
readingTime: 45
difficulty: 4
```

This powers the paper atlas, the lineage lens, timeline visualisations, and "what should I
read next" traversal — **without** a single paper summary that lacks a mechanism link.

## Copyright policy for content

- Never reproduce figures from papers. Redraw from the described mechanism, in our visual
  language, with attribution to the source idea.
- Quote sparingly and with citation; never paste abstracts wholesale.
- Model weights used for traces must be permissively licensed; the licence is recorded in the
  trace manifest and surfaced in the UI.
- Content licensed **CC BY-SA 4.0**, code **Apache-2.0**. Contributors retain copyright via DCO
  sign-off; no CLA (lowers the barrier and avoids the trust problem CLAs create).
