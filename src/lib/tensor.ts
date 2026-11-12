// Small, honest tensor helpers. Everything here runs for real in the browser —
// no fake numbers anywhere in the app.

export type Mat = number[][]

export function zeros(r: number, c: number): Mat {
  return Array.from({ length: r }, () => new Array(c).fill(0))
}

/** Deterministic RNG so every visitor sees the same numbers. */
export function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

/** Normal samples via Box-Muller, from a seeded uniform. */
export function randn(r: number, c: number, seed = 42, scale = 1): Mat {
  const u = rng(seed)
  const out = zeros(r, c)
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      const a = Math.max(u(), 1e-12), b = u()
      out[i][j] = Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b) * scale
    }
  return out
}

export function matmul(a: Mat, b: Mat): Mat {
  const n = a.length, k = b.length, m = b[0].length
  const out = zeros(n, m)
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const aip = a[i][p]
      if (aip === 0) continue
      for (let j = 0; j < m; j++) out[i][j] += aip * b[p][j]
    }
  return out
}

export function transpose(a: Mat): Mat {
  const out = zeros(a[0].length, a.length)
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[0].length; j++) out[j][i] = a[i][j]
  return out
}

export function scale(a: Mat, s: number): Mat {
  return a.map(r => r.map(v => v * s))
}

/**
 * Softmax over the last axis.
 * `subtractMax` exists so a lab can turn it OFF and show the overflow that follows.
 */
export function softmaxRows(a: Mat, subtractMax = true): Mat {
  return a.map(row => {
    const mx = subtractMax ? Math.max(...row) : 0
    const ex = row.map(v => Math.exp(v - mx))
    const sum = ex.reduce((x, y) => x + y, 0)
    return ex.map(v => v / sum)
  })
}

/** Causal mask: position i may only look at j <= i. */
export function causalMask(a: Mat): Mat {
  return a.map((row, i) => row.map((v, j) => (j <= i ? v : -Infinity)))
}

export function entropy(p: number[]): number {
  let h = 0
  for (const v of p) if (v > 1e-12) h -= v * Math.log2(v)
  return h
}

export function minMax(a: Mat): [number, number] {
  let lo = Infinity, hi = -Infinity
  for (const row of a) for (const v of row) {
    if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v }
  }
  if (lo === Infinity) { lo = 0; hi = 1 }
  return [lo, hi]
}

/** One attention head, computed step by step so a lab can show every stage. */
export function attentionHead(
  x: Mat, wq: Mat, wk: Mat, wv: Mat, opts: { causal?: boolean; scaled?: boolean } = {},
) {
  const { causal = true, scaled = true } = opts
  const q = matmul(x, wq)
  const k = matmul(x, wk)
  const v = matmul(x, wv)
  const dHead = q[0].length
  let scores = matmul(q, transpose(k))
  const rawScores = scores.map(r => [...r])
  if (scaled) scores = scale(scores, 1 / Math.sqrt(dHead))
  const masked = causal ? causalMask(scores) : scores
  const weights = softmaxRows(masked)
  const out = matmul(weights, v)
  return { q, k, v, rawScores, scores, masked, weights, out, dHead }
}
