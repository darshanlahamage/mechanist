# Contributing

The goal is for someone to finish a page and be able to **predict** what the thing does — not
recall that it exists.

## Start small

| Size | What | Time |
|---|---|---|
| Tiny | Fix a wrong number, a typo, a broken link | 5 min |
| Small | Add a row to a failure table from something you have actually hit | 20 min |
| Medium | Write one view of an existing concept — usually Systems or Failure | 2 h |
| Large | A new lab | 1 day |
| Largest | A whole concept, all six views | 2–3 days |

**Writing the Systems or Failure view of an existing concept is the most useful thing you can
do.** Most concepts will have intuition and math long before anyone writes down what the thing
costs or how it fails in production. That knowledge usually only exists in people's heads.

## Adding a concept

1. Create `src/content/concepts/<name>.tsx`:

```tsx
import type { Concept } from '../types'
import { Code, Eq, Note } from '../../components/ui'

export const myConcept: Concept = {
  id: 'my-concept',
  title: 'My concept',
  tagline: 'One sentence that says what it actually does.',
  track: 'transformer',
  difficulty: 3,
  status: 'stable',
  prereqs: ['attention'],
  papers: [{ title: '...', year: 2023, url: 'https://arxiv.org/abs/...',
             idea: 'The one thing this paper added.' }],
  lenses: {
    intuition: () => <>...</>,
    mechanism: () => <>...</>,
    math:      () => <>...</>,
    code:      () => <>...</>,
    systems:   () => <>...</>,
    failure:   () => <>...</>,
  },
}
```

2. Import it in `src/content/registry.ts` and replace the `soon(...)` placeholder.

Partial is fine. A concept with two good views beats a concept with six padded ones.

## Adding a lab

Labs live in `src/labs/`. Use the pieces in `src/components/ui.tsx`:

`Lab` `Panel` `Slider` `Select` `Toggle` `Stat` `Heat` `Bars` `Line` `Eq` `Code` `Note`

```tsx
export default function MyLab() {
  const [n, setN] = useState(8)
  const result = useMemo(() => realComputation(n), [n])

  return (
    <Lab title="What this shows" takeaway={<>The one thing to walk away with.</>}>
      <Panel title="Controls"><Slider label="n" min={1} max={64} value={n} onChange={setN} /></Panel>
      <Heat m={result} />
    </Lab>
  )
}
```

## Rules that are not negotiable

1. **Real computation only.** If a visual shows attention weights, they must come from an actual
   softmax over actual scores. Never draw a picture of what the math would look like.
2. **State shapes.** Every tensor's shape, where it matters.
3. **Say where numbers came from.** Derived, measured on named hardware, or cited. If it is an
   estimate, call it an estimate.
4. **Include a way to break it.** A toggle that produces the characteristic failure is worth more
   than three paragraphs describing it.
5. **Plain words.** Technical, but unfolded. Write "reads every weight out of memory", not
   "exhibits suboptimal memory access characteristics".
6. **No paper figures.** Redraw from the mechanism. Copying figures is a copyright problem.

## Checks before you open a PR

```bash
npx tsc -b --noEmit
npm run build
```

Open the page and use your own lab. If you cannot explain what it taught you in one sentence,
it is not finished.
