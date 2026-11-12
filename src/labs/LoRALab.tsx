import { useMemo, useState } from 'react'
import { frobenius, lowRankApprox, svd } from '../lib/linalg'
import { randn, zeros, type Mat } from '../lib/tensor'
import { Heat, Lab, Line, Panel, Select, Slider, Stat } from '../components/ui'

const M = 48, N = 48
type Kind = 'low rank (typical fine-tune)' | 'mixed' | 'full rank (noise)'

/** Build a weight update with a known structure, so the lesson is about rank, not luck. */
function makeDelta(kind: Kind): Mat {
  const out = zeros(M, N)
  const noise = randn(M, N, 5, 0.06)
  if (kind !== 'full rank (noise)') {
    const r = kind === 'low rank (typical fine-tune)' ? 3 : 10
    for (let c = 0; c < r; c++) {
      const u = randn(M, 1, 100 + c).map(x => x[0])
      const v = randn(1, N, 200 + c)[0]
      const s = 1 / (c + 1)
      for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) out[i][j] += s * u[i] * v[j]
    }
  }
  const mix = kind === 'full rank (noise)' ? 1 : kind === 'mixed' ? 0.5 : 0.15
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) out[i][j] += mix * noise[i][j]
  return out
}

export default function LoRALab() {
  const [rank, setRank] = useState(4)
  const [kind, setKind] = useState<Kind>('low rank (typical fine-tune)')

  const delta = useMemo(() => makeDelta(kind), [kind])
  const { U, S, V } = useMemo(() => svd(delta, 24), [delta])

  const approx = useMemo(() => lowRankApprox(U, S, V, rank, M, N), [U, S, V, rank])
  const residual = useMemo(() => delta.map((r, i) => r.map((v, j) => v - approx[i][j])), [delta, approx])

  const total = frobenius(delta)
  const err = frobenius(delta, approx)
  const captured = 1 - err / total

  const fullParams = M * N
  const loraParams = rank * (M + N)
  const spectrum = S.map((s, i) => [i, s] as [number, number])
  const energy: [number, number][] = []
  let acc = 0
  const sumSq = S.reduce((a, b) => a + b * b, 0)
  S.forEach((s, i) => { acc += s * s; energy.push([i + 1, acc / sumSq]) })

  return (
    <Lab title="What a low rank update can and cannot express"
      takeaway={<>LoRA does not approximate the weights. It approximates the <b>change</b> to the
        weights, and that change is usually close to low rank, which is why a rank of 8 or 16 works
        at all. Switch the update to full rank noise and watch the error refuse to fall no matter how
        high you push the rank — that is what happens when you try to use LoRA to teach genuinely new
        knowledge rather than a new behaviour.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Slider label="LoRA rank (r)" min={1} max={24} value={rank} onChange={setRank} />
              <Select label="what the update looks like" value={kind}
                      options={['low rank (typical fine-tune)', 'mixed', 'full rank (noise)'] as const}
                      onChange={setKind} />
            </div>
          </Panel>
          <Stat label="change captured" value={`${(captured * 100).toFixed(1)}%`}
                tone={captured > 0.9 ? 'good' : captured > 0.7 ? 'warn' : 'bad'} />
          <Stat label="trainable values" value={loraParams.toLocaleString()}
                sub={`${((loraParams / fullParams) * 100).toFixed(1)}% of ${fullParams.toLocaleString()}`} tone="good" />
          <Stat label="A and B shapes" value={`${M}×${rank}, ${rank}×${N}`} sub="the two LoRA matrices" />
        </div>

        <div className="space-y-4">
          <Panel title="The weight change, the rank-r version of it, and what is left over">
            <div className="flex flex-wrap gap-5">
              <Heat m={delta} cell={5} title="ΔW — the real change" />
              <Heat m={approx} cell={5} title={`B·A — rank ${rank}`} />
              <Heat m={residual} cell={5} title="what LoRA misses" />
            </div>
          </Panel>

          <div className="grid gap-3 lg:grid-cols-2">
            <Panel title="Singular values">
              <Line series={[{ name: 'σ', color: '#0b6f66', points: spectrum }]}
                    markers={[{ x: rank - 1, label: 'r' }]} xLabel="component" height={140} />
              <div className="mt-2 text-[12px] leading-relaxed text-[var(--dim)]">
                A steep drop means the change really is low rank, and a small r is enough.
                A flat line means every direction matters equally.
              </div>
            </Panel>
            <Panel title="Change captured by rank">
              <Line series={[{ name: 'captured', color: '#6a46a3', points: energy }]}
                    markers={[{ x: rank, label: 'you' }]} xLabel="rank" height={140}
                    yFmt={v => `${(v * 100).toFixed(0)}%`} />
              <div className="mt-2 text-[12px] leading-relaxed text-[var(--dim)]">
                Where this curve flattens is the rank worth paying for. Past that you are adding
                parameters that capture nothing.
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </Lab>
  )
}
