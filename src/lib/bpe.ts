// A real byte-pair encoding implementation. Small, but the same algorithm the
// big tokenizers use: count adjacent pairs, merge the most common, repeat.

export interface Merge { a: string; b: string; count: number; rank: number }

export function trainBPE(corpus: string, numMerges: number) {
  // start from characters, with a visible end-of-word marker
  let words: string[][] = corpus
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map(w => [...w, '</w>'])

  const merges: Merge[] = []
  const snapshots: string[][][] = [words.map(w => [...w])]

  for (let r = 0; r < numMerges; r++) {
    const counts = new Map<string, number>()
    for (const w of words)
      for (let i = 0; i < w.length - 1; i++) {
        const key = w[i] + '\u0000' + w[i + 1]
        counts.set(key, (counts.get(key) ?? 0) + 1)
      }
    if (counts.size === 0) break

    let best = '', bestN = 0
    for (const [k, n] of counts) if (n > bestN) { best = k; bestN = n }
    if (bestN < 2) break

    const [a, b] = best.split('\u0000')
    words = words.map(w => {
      const out: string[] = []
      for (let i = 0; i < w.length; i++) {
        if (i < w.length - 1 && w[i] === a && w[i + 1] === b) { out.push(a + b); i++ }
        else out.push(w[i])
      }
      return out
    })
    merges.push({ a, b, count: bestN, rank: r })
    snapshots.push(words.map(w => [...w]))
  }

  const vocab = new Set<string>()
  for (const w of words) for (const t of w) vocab.add(t)
  return { merges, snapshots, words, vocabSize: vocab.size }
}

/** Apply learned merges to new text, in rank order. This is encoding. */
export function encodeBPE(text: string, merges: Merge[]): string[][] {
  return text.toLowerCase().split(/\s+/).filter(Boolean).map(word => {
    let w = [...word, '</w>']
    for (const m of merges) {
      const out: string[] = []
      for (let i = 0; i < w.length; i++) {
        if (i < w.length - 1 && w[i] === m.a && w[i + 1] === m.b) { out.push(m.a + m.b); i++ }
        else out.push(w[i])
      }
      w = out
    }
    return w
  })
}

export const SAMPLE_CORPUS = `
the cat sat on the mat the cat ate the rat
a cat and a rat and a hat and a bat
low lower lowest slow slower slowest new newer newest
running jumping walking talking running jumping
the model learns the tokens the model learns the words
unhappy unhappily happiness happily happier unhappier
`.trim()
