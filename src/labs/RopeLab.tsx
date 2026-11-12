import { useMemo, useState } from 'react'
import { randn } from '../lib/tensor'
import { Lab, Line, Panel, Select, Slider, Stat } from '../components/ui'

type Scaling = 'none' | 'linear (PI)' | 'ntk-aware'

/** Rotate a vector by position, two dimensions at a time. This is all RoPE is. */
function rope(v: number[], pos: number, base: number): number[] {
  const d = v.length
  const out = new Array(d).fill(0)
  for (let i = 0; i < d / 2; i++) {
    const theta = Math.pow(base, (-2 * i) / d)
    const a = pos * theta
    const c = Math.cos(a), s = Math.sin(a)
    const x = v[2 * i], y = v[2 * i + 1]
    out[2 * i] = x * c - y * s
    out[2 * i + 1] = x * s + y * c
  }
  return out
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)

export default function RopeLab() {
  const [d, setD] = useState(64)
  const [base, setBase] = useState(10000)
  const [trainLen, setTrainLen] = useState(512)
  const [testLen, setTestLen] = useState(2048)
  const [scaling, setScaling] = useState<Scaling>('none')
  const [inspectPos, setInspectPos] = useState(128)

  const factor = Math.max(1, testLen / trainLen)
  const effBase = scaling === 'ntk-aware' ? base * Math.pow(factor, d / (d - 2)) : base
  const posScale = scaling === 'linear (PI)' ? 1 / factor : 1

  const { curve, baseline, angles, wavelengths } = useMemo(() => {
    const v = randn(1, d, 3)[0]
    const k0 = rope(v, 0, effBase)
    const curve: [number, number][] = []
    const baseline: [number, number][] = []
    const step = Math.max(1, Math.floor(testLen / 260))
    for (let p = 0; p <= testLen; p += step) {
      curve.push([p, dot(rope(v, p * posScale, effBase), k0) / d])
      baseline.push([p, dot(rope(v, p, base), rope(v, 0, base)) / d])
    }
    const angles: number[] = []
    const wavelengths: number[] = []
    for (let i = 0; i < d / 2; i++) {
      const theta = Math.pow(effBase, (-2 * i) / d)
      angles.push((inspectPos * posScale * theta) / (2 * Math.PI))
      wavelengths.push((2 * Math.PI) / theta)
    }
    return { curve, baseline, angles, wavelengths }
  }, [d, base, effBase, posScale, testLen, inspectPos])

  const fastTurns = angles[0]
  const slowTurns = angles[angles.length - 1]

  return (
    <Lab title="RoPE: position as rotation"
      takeaway={<>RoPE never adds anything to the vector — it rotates pairs of dimensions by an angle
        that grows with position. Because the rotation is shared, the dot product between two tokens
        depends only on the <b>distance between them</b>. Push past the trained length and the fastest
        dimensions have spun thousands of times into angles the model never saw. That is what context
        extension methods fix, by slowing the spin down.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Slider label="head dim (d)" min={8} max={256} step={8} value={d} onChange={setD} />
              <Slider label="base (θ)" min={1000} max={1000000} step={1000} value={base}
                      onChange={setBase} fmt={v => v.toLocaleString()} />
              <Slider label="trained length" min={128} max={4096} step={128} value={trainLen} onChange={setTrainLen} />
              <Slider label="tested length" min={256} max={32768} step={256} value={testLen} onChange={setTestLen} />
              <Select label="context extension" value={scaling}
                      options={['none', 'linear (PI)', 'ntk-aware'] as const} onChange={setScaling} />
              <Slider label="inspect position" min={0} max={testLen} step={8} value={Math.min(inspectPos, testLen)}
                      onChange={setInspectPos} />
            </div>
          </Panel>
          <Stat label="stretch factor" value={`${factor.toFixed(1)}×`}
                sub={scaling === 'none' ? 'nothing applied' : scaling === 'linear (PI)' ? 'positions squeezed' : `base → ${effBase.toExponential(1)}`} />
          <Stat label="fastest dim turns" value={fastTurns.toFixed(1)}
                sub={`at position ${inspectPos}`} tone={fastTurns > 200 ? 'warn' : 'normal'} />
          <Stat label="slowest dim turns" value={slowTurns.toFixed(4)} sub="barely moves — carries long range" />
        </div>

        <div className="space-y-4">
          <Panel title="Dot product between two identical vectors, as they move apart">
            <Line
              series={[
                { name: `with ${scaling}`, color: '#5eead4', points: curve },
                ...(scaling !== 'none' ? [{ name: 'no scaling', color: '#6b7483', points: baseline, dashed: true }] : []),
              ]}
              markers={[{ x: trainLen, label: 'trained to here' }]}
              xLabel="distance between tokens" yLabel="similarity" />
            <div className="mt-2 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Nearby tokens stay similar; far ones fall away and oscillate. Nothing told the model to
              do this — it comes out of the rotation itself.
            </div>
          </Panel>

          <Panel title={`Rotation at position ${inspectPos} — one dot per dimension pair`}>
            <Line series={[{ name: 'full turns', color: '#a78bfa',
                             points: angles.map((a, i) => [i, a] as [number, number]) }]}
                  xLabel="dimension pair (fast → slow)" yLabel="full turns" height={140} />
            <div className="mt-2 grid grid-cols-2 gap-2 text-[12px] text-[var(--dim)] sm:grid-cols-4">
              {[0, Math.floor(d / 8), Math.floor(d / 4), Math.floor(d / 2) - 1].map(i => (
                <div key={i} className="rounded border border-[var(--line)] px-2 py-1.5">
                  <div className="font-mono text-[11px] text-[#c7cdd8]">pair {i}</div>
                  <div className="font-mono text-[11px] text-[var(--accent)]">
                    wavelength {wavelengths[i] > 9999 ? wavelengths[i].toExponential(1) : Math.round(wavelengths[i])}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Early pairs spin fast and encode "right next to me". Late pairs spin so slowly their
              wavelength is longer than the whole context — they encode "roughly where in the document".
              Raising the base slows every pair down, which is why it buys longer context.
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
