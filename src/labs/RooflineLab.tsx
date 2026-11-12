import { useMemo, useState } from 'react'
import { BYTES, HARDWARE, PRESETS, type DType } from '../lib/cost'
import { Lab, Panel, Select, Slider, Stat } from '../components/ui'

/** Each entry computes its own FLOPs and bytes, so the point on the chart is derived, not drawn. */
const HEADS = 32

function operations(d: number, seq: number, batch: number, dt: DType) {
  const b = BYTES[dt]
  const ops = [
    { name: 'add two tensors', flops: seq * d, bytes: 3 * seq * d * b },
    { name: 'RMSNorm', flops: 4 * seq * d, bytes: 2 * seq * d * b },
    { name: 'softmax (row)', flops: 5 * seq * d, bytes: 2 * seq * d * b },
    { name: 'decode matvec', flops: 2 * d * d * batch, bytes: (d * d + 2 * d * batch) * b },
    { name: 'GEMM 512³', flops: 2 * 512 ** 3, bytes: 3 * 512 * 512 * b },
    { name: 'GEMM 4096³', flops: 2 * 4096 ** 3, bytes: 3 * 4096 * 4096 * b },
    // attention is per-head: the score matrix is heads x seq x seq, which is the
    // whole reason naive attention is memory bound and flash attention is not
    { name: 'attention prefill', flops: 4 * seq * seq * d,
      bytes: (3 * HEADS * seq * seq + 4 * seq * d) * b },
    { name: 'flash attention', flops: 4 * seq * seq * d, bytes: 4 * seq * d * b },
  ]
  return ops.map(o => ({ ...o, intensity: o.flops / o.bytes }))
}

export default function RooflineLab() {
  const [hwName, setHwName] = useState(HARDWARE[0].name)
  const [dt, setDt] = useState<DType>('bf16')
  const [seq, setSeq] = useState(4096)
  const [batch, setBatch] = useState(1)
  const d = PRESETS['Llama 3 8B'].dModel

  const hw = HARDWARE.find(h => h.name === hwName)!
  const peak = hw.flops[dt] ?? hw.flops.bf16 ?? hw.flops.fp16 ?? 1
  const ridge = peak / hw.bandwidth
  const ops = useMemo(() => operations(d, seq, batch, dt), [d, seq, batch, dt])

  const W = 560, H = 300, L = 56, B = 34, T = 14, R = 12
  const xMin = 0.05, xMax = 2000
  const yMin = peak / 3000, yMax = peak * 1.6
  const lx = (v: number) => L + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - L - R)
  const ly = (v: number) => T + (1 - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - T - B)
  const attain = (i: number) => Math.min(peak, i * hw.bandwidth)

  const fmtF = (v: number) => v >= 1e12 ? `${(v / 1e12).toFixed(0)}T` : v >= 1e9 ? `${(v / 1e9).toFixed(0)}G` : `${(v / 1e6).toFixed(0)}M`

  return (
    <Lab title="The roofline: is this operation waiting on math or on memory?"
      takeaway={<>Every operation has one number that decides its fate: how much arithmetic it does
        per byte it moves. Below the slanted roof you are memory bound and more FLOPs would not help.
        Under the flat roof you are compute bound. Notice flash attention and plain attention do the
        exact same arithmetic, and sit in completely different places — because one of them moves far
        fewer bytes.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Select label="hardware" value={hwName} options={HARDWARE.map(h => h.name)} onChange={setHwName} />
              <Select label="dtype" value={dt} options={['bf16', 'fp16', 'fp32', 'fp8'] as const} onChange={setDt} />
              <Slider label="sequence length" min={256} max={32768} step={256} value={seq} onChange={setSeq}
                      fmt={v => v >= 1024 ? `${(v / 1024).toFixed(0)}k` : String(v)} />
              <Slider label="batch" min={1} max={256} value={batch} onChange={setBatch} />
            </div>
          </Panel>
          <Stat label="ridge point" value={ridge.toFixed(0)}
                sub="FLOPs per byte needed to saturate" tone="good" />
          <Stat label="peak compute" value={`${(peak / 1e12).toFixed(0)} TFLOP/s`} sub={dt} />
          <Stat label="bandwidth" value={`${(hw.bandwidth / 1e9).toFixed(0)} GB/s`} sub={hw.note} />
        </div>

        <div className="space-y-4">
          <Panel title="Where each operation lands">
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
              {[1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15].filter(v => v > yMin && v < yMax).map(v => (
                <g key={v}>
                  <line x1={L} x2={W - R} y1={ly(v)} y2={ly(v)} stroke="#1f252e" />
                  <text x={L - 6} y={ly(v) + 3} textAnchor="end" fontSize="9" fill="#6b7483" fontFamily="ui-monospace">{fmtF(v)}</text>
                </g>
              ))}
              {[0.1, 1, 10, 100, 1000].map(v => (
                <g key={v}>
                  <line x1={lx(v)} x2={lx(v)} y1={T} y2={H - B} stroke="#1f252e" />
                  <text x={lx(v)} y={H - 20} textAnchor="middle" fontSize="9" fill="#6b7483" fontFamily="ui-monospace">{v}</text>
                </g>
              ))}
              {/* the roof itself */}
              <polyline fill="none" stroke="#5eead4" strokeWidth="2"
                points={`${lx(xMin)},${ly(Math.max(attain(xMin), yMin))} ${lx(ridge)},${ly(peak)} ${lx(xMax)},${ly(peak)}`} />
              <line x1={lx(ridge)} x2={lx(ridge)} y1={T} y2={H - B} stroke="#5eead4" strokeWidth="1" strokeDasharray="3 3" opacity="0.5" />
              <text x={lx(ridge) + 4} y={T + 10} fontSize="9" fill="#5eead4">ridge {ridge.toFixed(0)}</text>
              <text x={lx(0.12)} y={ly(attain(0.12)) - 8} fontSize="9" fill="#6b7483">memory bound</text>
              <text x={W - R - 90} y={ly(peak) - 8} fontSize="9" fill="#6b7483">compute bound</text>

              {ops.map(o => {
                const x = lx(Math.max(o.intensity, xMin)), y = ly(Math.max(attain(o.intensity), yMin))
                const mem = o.intensity < ridge
                return (
                  <g key={o.name}>
                    <circle cx={x} cy={y} r="4" fill={mem ? '#fbbf24' : '#a78bfa'} />
                    <text x={x + 7} y={y + 3} fontSize="9" fill="#c7cdd8">{o.name}</text>
                  </g>
                )
              })}
              <text x={W - R} y={H - 4} textAnchor="end" fontSize="9" fill="#6b7483">arithmetic intensity (FLOPs per byte)</text>
              <text x={4} y={12} fontSize="9" fill="#6b7483">FLOP/s reachable</text>
            </svg>
          </Panel>

          <Panel title="The numbers behind each point">
            <table className="w-full font-mono text-[11.5px]">
              <thead><tr className="text-[var(--dim)]">
                <th className="py-1 text-left font-normal">operation</th>
                <th className="py-1 text-right font-normal">FLOPs</th>
                <th className="py-1 text-right font-normal">bytes</th>
                <th className="py-1 text-right font-normal">per byte</th>
                <th className="py-1 text-right font-normal">bound by</th>
              </tr></thead>
              <tbody>
                {ops.map(o => (
                  <tr key={o.name} className="border-t border-[var(--line)]">
                    <td className="py-1 text-[#c7cdd8]">{o.name}</td>
                    <td className="py-1 text-right text-[var(--dim)]">{fmtF(o.flops)}</td>
                    <td className="py-1 text-right text-[var(--dim)]">{(o.bytes / 1e6).toFixed(1)}M</td>
                    <td className="py-1 text-right text-[var(--accent)]">{o.intensity.toFixed(1)}</td>
                    <td className={`py-1 text-right ${o.intensity < ridge ? 'text-[var(--warn)]' : 'text-[var(--accent-2)]'}`}>
                      {o.intensity < ridge ? 'memory' : 'compute'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
