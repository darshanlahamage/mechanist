import { useState } from 'react'
import { Bars, Code, Lab, Panel, Select, Slider, Stat, Toggle } from '../components/ui'

/** JavaScript only has float64. These narrow a value to what a smaller format could hold,
 *  so the overflow happens at the same logit value a real GPU would hit. */
const DTYPE = {
  float16: { cast: (v: number) => (Math.abs(v) > 65504 ? Math.sign(v) * Infinity : Math.fround(v)),
             max: 65504, limit: Math.log(65504) },
  float32: { cast: (v: number) => Math.fround(v), max: 3.4028235e38, limit: Math.log(3.4028235e38) },
  float64: { cast: (v: number) => v, max: 1.7976931e308, limit: Math.log(1.7976931e308) },
} as const
type DT = keyof typeof DTYPE

export default function SoftmaxLab() {
  const [scale, setScale] = useState(6)
  const [subtractMax, setSubtractMax] = useState(true)
  const [dtype, setDtype] = useState<DT>('float32')

  const { cast, limit } = DTYPE[dtype]
  const logits = [0.8, -0.4, 1.9, 0.1, -1.2, 2.4].map(v => cast(v * scale))
  const mx = subtractMax ? Math.max(...logits) : 0
  const exps = logits.map(v => cast(Math.exp(v - mx)))
  const sum = cast(exps.reduce((a, b) => cast(a + b), 0))
  const probs = exps.map(e => cast(e / sum))
  const broken = probs.some(p => !Number.isFinite(p))
  const biggestLogit = Math.max(...logits)
  const headroom = limit - (biggestLogit - mx)

  return (
    <Lab title="Why softmax subtracts the max"
      takeaway={<>Subtracting the max changes nothing mathematically — the ratio is identical.
        It changes everything numerically. Switch to float16 and the whole thing falls over at a
        logit of about 11, which real models pass routinely. This is why attention and softmax
        accumulate in float32 even when the weights are half precision, and why online softmax in
        Flash Attention is this same trick applied one tile at a time.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Slider label="logit size" min={1} max={140} value={scale} onChange={setScale} fmt={v => `×${v}`} />
              <Select label="number format" value={dtype}
                      options={['float16', 'float32', 'float64'] as const} onChange={setDtype} />
              <Toggle label="subtract max first" value={subtractMax} onChange={setSubtractMax}
                      hint={subtractMax ? 'exp(x − max)' : 'exp(x) directly'} />
            </div>
          </Panel>
          <Stat label="exp overflows above" value={limit.toFixed(1)}
                sub={`largest ${dtype} value is ${DTYPE[dtype].max.toExponential(1)}`} />
          <Stat label="biggest exponent used" value={(biggestLogit - mx).toFixed(1)}
                sub={subtractMax ? 'always 0 after subtracting' : 'the raw logit'}
                tone={headroom < 0 ? 'bad' : headroom < 10 ? 'warn' : 'good'} />
          <Stat label="probabilities" value={broken ? 'NaN' : 'valid'} tone={broken ? 'bad' : 'good'}
                sub={broken ? 'Infinity / Infinity' : `sum = ${probs.reduce((a, b) => a + b, 0).toFixed(6)}`} />
        </div>

        <div className="space-y-4">
          <Panel title="The numbers at each step">
            <div className="overflow-x-auto">
              <table className="w-full font-mono text-[12px]">
                <thead>
                  <tr className="text-[var(--dim)]">
                    <th className="py-1 text-left font-normal">logit</th>
                    <th className="py-1 text-right font-normal">{subtractMax ? 'x − max' : 'x'}</th>
                    <th className="py-1 text-right font-normal">exp(·)</th>
                    <th className="py-1 text-right font-normal">prob</th>
                  </tr>
                </thead>
                <tbody>
                  {logits.map((l, i) => (
                    <tr key={i} className="border-t border-[var(--line)]">
                      <td className="py-1 text-[#3e392f]">{l.toFixed(1)}</td>
                      <td className="py-1 text-right text-[#3e392f]">{(l - mx).toFixed(1)}</td>
                      <td className={`py-1 text-right ${Number.isFinite(exps[i]) ? 'text-[var(--accent)]' : 'text-[var(--bad)]'}`}>
                        {Number.isFinite(exps[i]) ? exps[i].toExponential(2) : 'Infinity'}
                      </td>
                      <td className={`py-1 text-right ${Number.isFinite(probs[i]) ? 'text-[#3e392f]' : 'text-[var(--bad)]'}`}>
                        {Number.isFinite(probs[i]) ? probs[i].toFixed(4) : 'NaN'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          {!broken && <Panel title="Resulting distribution"><Bars values={probs} /></Panel>}
          {broken && (
            <div className="rounded-lg border-l-2 border-[var(--bad)] bg-[rgba(176,58,49,0.07)] px-4 py-3 text-[13px] leading-relaxed text-[#3e392f]">
              <b className="text-[var(--bad)]">This is the bug.</b> exp() passed the largest number
              {' '}{dtype} can hold, so it became Infinity. Dividing Infinity by Infinity gives NaN.
              One NaN here spreads through the whole network in a single step, and your loss prints
              <code> nan</code> from then on.
            </div>
          )}

          <Code lang="python">{`# the only difference between a working softmax and a broken one
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)   # <- this line
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)`}</Code>
        </div>
      </div>
    </Lab>
  )
}
