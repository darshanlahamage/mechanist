import { useMemo, useState } from 'react'
import { Lab, Panel, Select, Slider, Stat } from '../components/ui'

type Fmt = 'float32' | 'float16' | 'bfloat16' | 'fp8 (e4m3)'

const SPEC: Record<Fmt, { exp: number; man: number; bias: number }> = {
  'float32':     { exp: 8,  man: 23, bias: 127 },
  'float16':     { exp: 5,  man: 10, bias: 15 },
  'bfloat16':    { exp: 8,  man: 7,  bias: 127 },
  'fp8 (e4m3)':  { exp: 4,  man: 3,  bias: 7 },
}

/** Round a float64 to the nearest value the chosen format can hold, and return its bits. */
function encode(v: number, fmt: Fmt) {
  const { exp, man, bias } = SPEC[fmt]
  if (v === 0) return { bits: '0'.repeat(1 + exp + man), value: 0, exponent: 0, overflow: false }

  const sign = v < 0 ? 1 : 0
  const a = Math.abs(v)
  let e = Math.floor(Math.log2(a))
  let mantissa = a / 2 ** e - 1                       // in [0, 1)
  let mBits = Math.round(mantissa * 2 ** man)
  if (mBits === 2 ** man) { mBits = 0; e += 1 }       // rounding carried into the exponent

  const maxExp = 2 ** exp - 1 - bias
  const overflow = e > maxExp
  const stored = overflow ? Infinity : (sign ? -1 : 1) * (1 + mBits / 2 ** man) * 2 ** e

  const eField = Math.max(0, Math.min(2 ** exp - 1, e + bias))
  const bits = `${sign}${eField.toString(2).padStart(exp, '0')}${mBits.toString(2).padStart(man, '0')}`
  return { bits, value: stored, exponent: e, overflow }
}

/** Gap to the next representable number at this magnitude. */
function gapAt(v: number, fmt: Fmt) {
  const { man } = SPEC[fmt]
  const e = Math.floor(Math.log2(Math.abs(v) || 1e-30))
  return 2 ** (e - man)
}

export default function FloatLab() {
  const [fmt, setFmt] = useState<Fmt>('float16')
  const [expSlider, setExpSlider] = useState(0)
  const [mantSlider, setMantSlider] = useState(0.5)

  const value = useMemo(() => (1 + mantSlider) * 2 ** expSlider, [expSlider, mantSlider])
  const enc = encode(value, fmt)
  const { exp, man } = SPEC[fmt]
  const relErr = enc.overflow ? Infinity : Math.abs(enc.value - value) / Math.abs(value)
  const gap = gapAt(value, fmt)

  const compare = (Object.keys(SPEC) as Fmt[]).map(f => {
    const e = encode(value, f)
    return { f, ...e, gap: gapAt(value, f),
             maxVal: (2 - 2 ** -SPEC[f].man) * 2 ** (2 ** SPEC[f].exp - 2 - SPEC[f].bias) }
  })

  return (
    <Lab title="What a float can and cannot hold"
      takeaway={<>Exponent bits buy you <b>range</b>. Mantissa bits buy you <b>precision</b>.
        bfloat16 and float16 are both 16 bits, but bfloat16 spends them on range and float16 spends
        them on precision. That single difference is why training moved to bfloat16: gradients span
        a huge range of magnitudes, and running out of range gives you infinity, while running out
        of precision only gives you a slightly wrong number.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Select label="format" value={fmt}
                      options={['float32', 'float16', 'bfloat16', 'fp8 (e4m3)'] as const} onChange={setFmt} />
              <Slider label="exponent" min={-30} max={40} value={expSlider} onChange={setExpSlider}
                      fmt={v => `2^${v}`} />
              <Slider label="mantissa" min={0} max={0.999} step={0.001} value={mantSlider}
                      onChange={setMantSlider} fmt={v => (1 + v).toFixed(3)} />
            </div>
          </Panel>
          <Stat label="you wanted" value={value.toExponential(4)} />
          <Stat label="you get" value={enc.overflow ? 'Infinity' : enc.value.toExponential(4)}
                tone={enc.overflow ? 'bad' : relErr > 0.01 ? 'warn' : 'good'} />
          <Stat label="relative error" value={enc.overflow ? '—' : `${(relErr * 100).toFixed(4)}%`} />
          <Stat label="gap to next number" value={gap.toExponential(2)}
                sub="nothing between these two exists" />
        </div>

        <div className="space-y-4">
          <Panel title={`${fmt} — ${1 + exp + man} bits`}>
            <div className="flex flex-wrap gap-[3px]">
              {enc.bits.split('').map((b, i) => {
                const kind = i === 0 ? 'sign' : i <= exp ? 'exp' : 'man'
                const col = kind === 'sign' ? '#f87171' : kind === 'exp' ? '#fbbf24' : '#5eead4'
                return (
                  <div key={i} className="grid h-7 w-6 place-items-center rounded font-mono text-[12px] font-bold"
                       style={{ background: b === '1' ? col : '#1a1f27',
                                color: b === '1' ? '#0b0d10' : '#4c5563' }}>{b}</div>
                )
              })}
            </div>
            <div className="mt-2.5 flex gap-4 text-[11.5px]">
              <span className="text-[#f87171]">1 sign</span>
              <span className="text-[#fbbf24]">{exp} exponent — range</span>
              <span className="text-[var(--accent)]">{man} mantissa — precision</span>
            </div>
            {enc.overflow && (
              <div className="mt-3 rounded-lg border-l-2 border-[var(--bad)] bg-[rgba(248,113,113,0.07)] px-3 py-2 text-[12.5px] text-[#c7cdd8]">
                Out of range for {fmt}. The exponent field saturated, so the value became Infinity.
                Every later calculation with it is garbage.
              </div>
            )}
          </Panel>

          <Panel title="The same number in every format">
            <table className="w-full font-mono text-[12px]">
              <thead><tr className="text-[var(--dim)]">
                <th className="py-1 text-left font-normal">format</th>
                <th className="py-1 text-right font-normal">stores</th>
                <th className="py-1 text-right font-normal">gap here</th>
                <th className="py-1 text-right font-normal">largest value</th>
              </tr></thead>
              <tbody>
                {compare.map(c => (
                  <tr key={c.f} className={`border-t border-[var(--line)] ${c.f === fmt ? 'bg-white/[0.03]' : ''}`}>
                    <td className="py-1.5 text-[#c7cdd8]">{c.f}</td>
                    <td className={`py-1.5 text-right ${c.overflow ? 'text-[var(--bad)]' : 'text-[var(--accent)]'}`}>
                      {c.overflow ? 'Infinity' : c.value.toExponential(3)}
                    </td>
                    <td className="py-1.5 text-right text-[var(--dim)]">{c.gap.toExponential(1)}</td>
                    <td className="py-1.5 text-right text-[var(--dim)]">{c.maxVal.toExponential(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-3 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Push the exponent past 15 and float16 gives up while bfloat16 carries on, even though
              both are 16 bits. Push the mantissa around at a fixed exponent and you will see
              bfloat16 is the coarser of the two.
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
