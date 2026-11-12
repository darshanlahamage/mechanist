# Mechanist

**See how AI models actually work.**

Most learning material tells you what a concept is called. This shows you what happens to the
numbers, why the formula has that shape, what it costs to run, and how it breaks.

Every visual here is real math running in your browser. No drawn animations, no made-up values.
Turn off the `1/√d` in attention and the softmax really does collapse. Switch the softmax lab to
float16 and it really does overflow at a logit of 11, same as it would on a GPU.

> Early and open. The concept map is large on purpose — it is the plan, in public.
> Grey entries in the sidebar are not written yet. Pick one and write it.

## Run it

```bash
npm install
npm run dev
```

Open http://localhost:5173

## What is in it so far

| Concept | What the lab lets you do |
|---|---|
| Scaled dot-product attention | Change head size, turn off `√d`, watch entropy collapse |
| RoPE | Drag positions, see the similarity curve, push past the trained length |
| Softmax and floating point | Switch between fp16/fp32/fp64 and find the exact overflow point |
| Diffusion forward process | Noise a real image, then watch error amplification wreck the reconstruction |
| Why generating tokens is slow | Pick a model and GPU, see memory-bound decode and where batching stops helping |

Around 100 more concepts are mapped across 14 tracks: foundations, network building blocks,
transformers, modern LLM architectures, training at scale, post-training, vision, multimodal,
diffusion, retrieval, GPUs and kernels, inference and serving, production, interpretability.

## How a concept is built

Every concept gets six views, because one view is never enough:

| View | Question it answers |
|---|---|
| Intuition | What is the idea? |
| Mechanism | What happens to the numbers? |
| Math | Why is the formula that shape? |
| Code | How do I write it? |
| Systems | What does it cost to run? |
| Failure | How does it break? |

The last two are the ones everyone else leaves out, and they are where most of the real
understanding is.

## Layout

```
src/
├── lib/           # tensor math and the shared cost model
├── components/    # UI pieces every lab is built from
├── labs/          # interactive labs
└── content/
    ├── registry.ts    # all tracks and concepts
    └── concepts/      # one file per written concept
docs/              # longer design notes and the full concept map
```

## Contributing

Pick any grey concept in the sidebar and write it. See [CONTRIBUTING.md](CONTRIBUTING.md).

The short version: a concept is one file in `src/content/concepts/`, and a lab is one file in
`src/labs/`. The components in `src/components/ui.tsx` do the plumbing, so a lab is usually
around 100 lines of actual idea.

## Licence

Code is Apache 2.0. Written content is CC BY-SA 4.0. Free, permanently.
