# 05 — Roadmap, Risks & Success Criteria

Phases are ordered so that **something excellent exists early**. A sprawling half-built atlas
attracts nobody; three perfect concepts attract contributors who then build the atlas.

---

## Phase 0 — Vertical Slice (weeks 1–3)
**Goal: prove the format end to end on three concepts. Nothing public yet.**

Build:
- Monorepo, Astro site, CI, design system skeleton
- `content-schema` (Zod) + graph builder + authoring CLI
- `lab-kit` v0: `Lab`, `ParamPanel`, `Stepper`, `TensorView`, `CodeSync`, `Predict`
- `engine-cost` v0: FLOPs/bytes/memory/roofline for attention + matmul + FFN
- `mechanist-trace` v0 and the `.mtr` format; capture GPT-2 small
- `mechanist-ref` v0: from-scratch attention, matmul, softmax, with golden tests vs PyTorch

Three concepts, complete with all six lenses:
1. **Scaled dot-product attention** — proves the Mechanism/Math lenses and trace playback
2. **Tiled matmul on GPU** — proves the systems/simulation path and the roofline engine
3. **KV cache & decode arithmetic** — proves the cost engine and the "numbers" thesis

**Exit criterion:** an ML engineer reads those three pages and says "I understood something I
did not understand before." Test with 5 real engineers before proceeding. If they do not say
that, fix the format — do not add content.

## Phase 1 — Transformer Track & Public Launch (weeks 4–12)
**Goal: the best transformer explanation on the internet, and a reason to contribute.**

- Track 3 complete: tokenisation → embeddings → attention variants → RoPE → FFN → blocks →
  full forward pass → decoding (~35 concepts)
- 🔥 **Forward Pass Theatre** and 🔥 **Attention Microscope** polished to flagship quality
- Track 0 and Track 1 to the depth the transformer track needs (~30 concepts)
- Paper atlas v1 (~90 papers) with the lineage view and timeline
- Search, graph navigation, learning paths v1, progress tracking (local-first)
- Contributor path: `CONTRIBUTING.md`, lab-kit docs, "your first concept in 90 minutes" guide,
  15 good-first-issue concept stubs with outlines already written
- **Public launch.** Target: HN front page, r/MachineLearning, ML Twitter/X, Discord opened.

**Exit criterion:** 5+ external contributors have merged a concept or lab without hand-holding.

## Phase 2 — Silicon (weeks 13–24)
**Goal: the track nobody else has. This is the long-term differentiator.**

- Track 11 core: memory hierarchy, roofline lab, coalescing, occupancy, the **Matmul Ladder**
  (8 rungs, measured on a real GPU in CI and published as artefacts)
- Triton track + **Triton Playground** on the Tier-3 runner (with hard quotas and caching)
- **Flash Attention Tiler** and the online-softmax derivation
- Quantisation deep dive + **Quantisation Scope**
- torch.compile / fusion inspector; CUDA graphs; profiling literacy
- TPU/systolic array animator; Apple Metal + MLX (makes the platform useful to every
  laptop owner, not just GPU renters)
- `mechanist-bench` producing versioned, citable measurements

**Exit criterion:** "kernel golf" leaderboard has real entries; the matmul ladder is being
linked to as a reference.

## Phase 3 — Breadth (weeks 25–44)
- Track 4 (modern LLM architectures: MoE, SSM/Mamba, MLA, long context, reasoning)
- Track 5 (training at scale) with the **Parallelism Simulator** and **Scaling Law Calculator**
- Track 12 (inference & serving) with the **Serving Simulator** and **Paged KV Visualiser**
- Track 9 (diffusion, full depth) with the **Diffusion Steppers** and flow-matching unification
- Track 7 + 8 (vision & multimodal) with the **ViT Dissector** and **Connector Lab**
- Track 6 (post-training, incl. the reward-hacking sandbox)

## Phase 4 — Depth & Durability (weeks 45+)
- Track 10 (retrieval), Track 13 (production + case studies), Track 14 (interpretability)
- Assessment engine + spaced repetition at full strength
- i18n (Hindi, Mandarin, Spanish, Korean first — where the audience actually is)
- Offline bundles, self-host docs, classroom/embed mode for teachers
- "Paper of the week" pipeline: every notable paper becomes a graph node within 2 weeks,
  which is what keeps the platform alive rather than a 2026 time capsule

---

## Risks and how each is actually handled

| Risk | Severity | Mitigation |
|---|---|---|
| **Scope explosion** — 420 concepts is a decade of solo work | Critical | Vertical slice first; strict concept template; `draft` status is publishable so partial work has value; the community builds breadth, maintainers build the *engine* |
| **Wrong content damages trust** | Critical | Verified-claims system; golden numerical tests in CI; domain CODEOWNERS; visible `lastVerified`; a public errata log with credit to reporters |
| **Content goes stale** (field moves monthly) | High | Staleness is a tracked, visible state; dependency-triggered `needs-update`; paper-of-the-week pipeline; design the graph so a new method is a node, not a rewrite |
| **Contributor drop-off after launch buzz** | High | Lab-kit so a lab is 100 lines; pre-written concept outlines as good-first-issues; fast review SLA (72h); public contributor credit on every page; a monthly release post naming people |
| **GPU runner cost spirals** | Medium | Tier 3 is optional by design; content-addressed cache; hard quotas; BYO-GPU token; sponsor credits (NVIDIA/Modal/Lambda all fund education); kill switch degrades to pre-measured results |
| **WebGPU unavailable or slow** | Medium | Every Tier-2 lab declares a Tier-1 fallback; CI tests the fallback path; no lab may render a blank box |
| **Maintainer burnout (you)** | High | Governance from day one; CODEOWNERS per track; no single-maintainer merge requirement; release trains not continuous heroics; explicit "this project survives me" bus-factor doc |
| **Paper figure / weights copyright** | Medium | Redraw-only policy enforced in review; permissive-licence-only weights; licence recorded in trace manifests |
| **"Yet another ML course" perception** | Medium | Lead with the two things nobody else has: real-tensor execution and the silicon track. The matmul ladder and Attention Microscope are the launch assets, not the homepage copy |
| **Accessibility as an afterthought** | Medium | a11y is a merge gate with a rubric item, not a backlog ticket; every viz has a data-table equivalent from day one |
| **Performance regression as labs accumulate** | Low | Per-lab JS budget enforced in CI; islands architecture means pages cannot bloat each other |

---

## Success criteria

Deliberately **not** pageviews. The metric must track understanding.

| Horizon | Measure | Target |
|---|---|---|
| Phase 0 | Engineers who report a genuine insight from the 3 slice pages | 4 / 5 |
| Phase 1 | Concepts at `stable` | 60 |
| Phase 1 | External contributors with a merged concept/lab | 5+ |
| Phase 1 | Median assessment accuracy *after* a concept, for people who failed it before | > 70% |
| Phase 2 | Monthly active learners completing ≥1 lab | 5,000 |
| Phase 2 | Inbound links from course syllabi / lab onboarding docs | 10 |
| Phase 3 | Contributors with ≥2 merged concepts | 25 |
| Phase 3 | Concepts at `stable` | 250 |
| Phase 4 | Claims verified by CI | > 90% |
| Always | Median time from a notable paper to a graph node | < 14 days |
| Always | Review turnaround on a first-time contribution | < 72 h |

## What "done" never means

The field will not stop. The platform's job is to make *new* things fast to absorb — a new
architecture should take an afternoon to add as a node and a day to understand, because every
prerequisite is already there and already visual. That is the real deliverable: not 420 pages,
but a **structure that makes the 421st concept cheap**.
