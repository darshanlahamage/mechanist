# 06 — Community, Governance & Sustainability

The platform cannot be built by one person. It also cannot be built by an unmoderated crowd —
ML education is full of confidently wrong material. This is the structure that lets volume and
quality coexist.

## Contribution ladder

Each rung is a real, valued contribution. Nobody is asked to start at the top.

| Rung | Contribution | Time | Review needed |
|---|---|---|---|
| 1 | Typo, clarity fix, broken link | 5 min | 1 approval, auto-merge on green |
| 2 | Worked example, misconception note, glossary entry | 30 min | 1 approval |
| 3 | Assessment items (predict / find-the-bug / diagnose) | 1 h | 1 domain approval |
| 4 | Paper entry (with the lineage fields filled honestly) | 1 h | 1 domain approval |
| 5 | A lens on an existing concept (usually Systems or Failure — most often missing) | 3 h | 1 domain + 1 pedagogy |
| 6 | A lab | 1 day | 1 domain + 1 pedagogy + perf/a11y gate |
| 7 | A full concept | 2–3 days | 1 domain + 1 pedagogy + all gates |
| 8 | A track (own a domain) | ongoing | Maintainer invitation |
| 9 | Core engine (lab-kit, cost engine, simulators, runner) | ongoing | Core team |

**Rung 5 is the secret weapon.** Hundreds of concepts will exist with Mechanism and Math but no
Systems or Failure lens. Those are small, well-scoped, high-value tasks that a working engineer
can do from their own experience — exactly the knowledge that is otherwise never written down.

## Review rubric (published, applied to every content PR)

1. **Correctness** — is the mechanism right? Are the numbers verified or honestly labelled?
2. **Mechanism over assertion** — does it show *why*, or just state *what*?
3. **Shape discipline** — is every tensor's shape stated where it matters?
4. **Honesty** — are the limits, the hardware dependence and the disagreements stated?
5. **Pedagogy** — does it predict and correct the common misconception? Does it ask the learner
   to predict before revealing?
6. **Prerequisites** — correctly declared, with no hidden assumed knowledge?
7. **Accessibility** — keyboard, non-visual equivalent, contrast, motion-reduction.
8. **Performance** — within the lab budget.

Reviewers quote the rubric item number. Authors know what to fix. This keeps review from being
taste-based arguments, which is what kills contributor morale.

## Governance

- **Core team** (3–7 people): engine, schema, release, final say on scope.
- **Track owners** (CODEOWNERS per track): review authority over their domain; invited after
  sustained contribution, not appointed.
- **Concept RFC** for anything that changes the schema, adds a track, or alters the
  pedagogical contract. Short template, public discussion, core-team decision, recorded as an ADR.
- **Decisions are logged.** `docs/adr/` holds every architectural and pedagogical decision with
  its context and consequences, so new contributors inherit reasoning instead of folklore.
- **Release train**, monthly: a changelog post naming every contributor, new concepts, new
  labs, corrections made. Predictability beats heroics.
- **Code of Conduct**: Contributor Covenant, enforced. A learning space for beginners that
  tolerates condescension is not a learning space.

## Recognition (the actual currency of OSS)

- Author and reviewer bylines on every concept page, linked to their profile.
- A contributors page with concept counts, and a graph of who built what.
- Monthly release post naming people and what they shipped.
- "Maintainer of \<track\>" is a real, citable credential — it is a stronger portfolio artefact
  than most blog posts, and that is a fair trade for the work.
- Conference talk / workshop material generated from the platform, co-presented with track owners.

## Sustainability

**Costs are deliberately near-zero by architecture.** Static site on a free/cheap CDN; traces on
R2 (no egress fees); Tier 3 is the only real expense and it is optional and cached.

| Source | Notes |
|---|---|
| GitHub Sponsors / Open Collective | Transparent ledger; funds runner GPU time first |
| Compute credits | NVIDIA, Modal, Lambda, RunPod, Cloudflare all run education programmes |
| Institutional support | University labs and companies that use it for onboarding |
| Paid *services*, never paid content | Workshops, team onboarding programmes, consulting. **All content stays free forever** — that promise is in the README and is non-negotiable |

Explicitly rejected: ads, paywalled concepts, "pro tier" for labs, selling learner data,
certification mills. Any of these would break the reason the community contributes.

## Adoption strategy

1. **Launch on the flagships, not the mission statement.** The Matmul Ladder and the Attention
   Microscope are the posts. The platform is what people find after.
2. **Be the thing people link to.** Optimise each concept page to be the best single answer to
   a question engineers actually ask ("why √d?", "what does GQA save?", "why is decode slow?").
3. **Courses and onboarding.** Offer embeddable labs free to universities and to company ML
   onboarding programmes; those become the durable traffic and contributor source.
4. **Paper-of-the-week.** Fast, visible, repeatable; brings researchers in as contributors.
5. **Kernel golf leaderboard.** Competitive, social, and it teaches the hardest track.
6. **Translation drives.** Each language community is a contributor community.
