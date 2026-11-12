import { type Mat, matmul, transpose, zeros } from './tensor'

/** Power iteration with deflation. Enough for the small matrices the labs use,
 *  and it lets you watch rank-k approximation improve one component at a time. */
export function svd(a: Mat, k: number) {
  const m = a.length, n = a[0].length
  let work = a.map(r => [...r])
  const U: number[][] = [], S: number[] = [], V: number[][] = []

  for (let c = 0; c < k; c++) {
    let v = new Array(n).fill(0).map((_, i) => Math.sin(i * 12.9898 + c * 7.233))
    let norm = Math.hypot(...v); v = v.map(x => x / norm)

    for (let it = 0; it < 120; it++) {
      // u = A v
      const u = work.map(row => row.reduce((s, x, j) => s + x * v[j], 0))
      const un = Math.hypot(...u) || 1e-12
      const uh = u.map(x => x / un)
      // v = A^T u
      const nv = new Array(n).fill(0)
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) nv[j] += work[i][j] * uh[i]
      const vn = Math.hypot(...nv) || 1e-12
      v = nv.map(x => x / vn)
    }
    const u = work.map(row => row.reduce((s, x, j) => s + x * v[j], 0))
    const sigma = Math.hypot(...u)
    const uh = u.map(x => x / (sigma || 1e-12))
    U.push(uh); S.push(sigma); V.push(v)
    // remove this component so the next iteration finds the next one
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) work[i][j] -= sigma * uh[i] * v[j]
  }
  return { U, S, V }
}

/** Rebuild the matrix from its first r components. This is exactly what LoRA can express. */
export function lowRankApprox(U: number[][], S: number[], V: number[][], r: number, m: number, n: number): Mat {
  const out = zeros(m, n)
  for (let c = 0; c < Math.min(r, S.length); c++)
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) out[i][j] += S[c] * U[c][i] * V[c][j]
  return out
}

export function frobenius(a: Mat, b?: Mat): number {
  let s = 0
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[0].length; j++) { const d = b ? a[i][j] - b[i][j] : a[i][j]; s += d * d }
  return Math.sqrt(s)
}

export { matmul, transpose }
