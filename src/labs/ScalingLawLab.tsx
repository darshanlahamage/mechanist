import { useMemo, useState } from 'react'
import { HARDWARE } from '../lib/cost'
import { Lab, Line, Note, Panel, Select, Slider, Stat } from '../components/ui'

// Chinchilla's fitted form: L(N, D) = E + A/N^alpha + B/D^beta
const E = 1.69, A = 406.4, ALPHA = 0.3392, Bc = 410.7, BETA = 0.2849
const loss = (N: number, D: number) => E + A / N ** ALPHA + Bc / D ** BETA

export default function ScalingLawLab() {
  const [logC, setLogC] = useState(22)         // training FLOPs, log10
  const [hwName, setHwName] = useState('H100 SXM')
  const [mfu, setMfu] = useState(0.4)
  const [perHour, setPerHour] = useState(2.5)

  const C = 10 ** logC
  const hw = HARDWARE.find(h => h.name === hwName)!
  const peak = hw.flops.bf16 ?? hw.flops.fp16 ?? 1e14

  const { curve, best, ratio } = useMemo(() => {
    const curve: [number, number][] = []
    let best = { N: 0, D: 0, L: Infinity }
    for (let ln = 7; ln <= 12.5; ln += 0.04) {
      const N = 10 ** ln
      const D = C / (6 * N)
      if (D < 1e8) continue
      const L = loss(N, D)
      curve.push([ln, L])
      if (L < best.L) best = { N, D, L }
    }
    return { curve, best, ratio: best.D / best.N }
  }, [C])

  const gpuHours = C / (peak * mfu) / 3600
  const cost = gpuHours * perHour
  const days1024 = gpuHours / 1024 / 24

  // what you get if you deliberately train a smaller model on more data
  const smallN = best.N / 3
  const smallD = C / (6 * smallN)
  const smallL = loss(smallN, smallD)

  const fmtN = (n: number) => (n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : (n / 1e6).toFixed(0) + 'M')
  const money = (v: number) => (v < 1e6
    ? '$' + v.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : '$' + (v / 1e6).toFixed(2) + 'M')

  return (
    <Lab title="Given this much compute, how big should the model be?"
      takeaway={<>For a fixed budget there is one best split between model size and data, and it is
        not "as big as possible". Chinchilla's answer was roughly <b>20 tokens per parameter</b>.
        But the curve near the bottom is almost flat, which is the useful part: you can train a model
        three times smaller on three times more data for almost the same loss, and then pay far less
        every single time you run it. That is why deployed models are usually trained well past the
        compute-optimal point.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Budget">
            <div className="space-y-3">
              <Slider label="training compute" min={18} max={26} step={0.1} value={logC}
                      onChange={setLogC} fmt={v => '10^' + v.toFixed(1) + ' FLOPs'} />
              <Select label="GPU" value={hwName} options={HARDWARE.map(h => h.name)} onChange={setHwName} />
              <Slider label="MFU" min={0.1} max={0.65} step={0.01} value={mfu}
                      onChange={setMfu} fmt={v => (v * 100).toFixed(0) + '%'} />
              <Slider label="price per GPU-hour" min={0.5} max={10} step={0.1} value={perHour}
                      onChange={setPerHour} fmt={v => '$' + v.toFixed(2)} />
            </div>
          </Panel>
          <Stat label="best model size" value={fmtN(best.N)} tone="good" />
          <Stat label="tokens to train on" value={(best.D / 1e9).toFixed(0) + 'B'}
                sub={ratio.toFixed(1) + ' tokens per parameter'} />
          <Stat label="predicted loss" value={best.L.toFixed(3)} />
        </div>

        <div className="space-y-4">
          <Panel title="Loss across every way of spending the same compute">
            <Line series={[{ name: 'predicted loss', color: '#0b6f66', points: curve }]}
                  markers={[{ x: Math.log10(best.N), label: 'best' }]}
                  xLabel="log10(parameters)" yLabel="loss" height={190} />
            <div className="mt-2 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Left side: model too small, it cannot use the data. Right side: model too big, it runs
              out of tokens before it learns. The bottom is broad, not sharp — that flatness is what
              lets you trade training cost for inference cost.
            </div>
          </Panel>

          <div className="grid gap-3 sm:grid-cols-2">
            <Panel title="What the run costs">
              <div className="space-y-2">
                <Stat label="GPU-hours" value={gpuHours < 1e6
                  ? gpuHours.toLocaleString(undefined, { maximumFractionDigits: 0 })
                  : (gpuHours / 1e6).toFixed(2) + 'M'} />
                <Stat label="money" value={money(cost)} tone="warn" />
                <Stat label="on 1024 GPUs" value={days1024.toFixed(1) + ' days'} />
              </div>
            </Panel>
            <Panel title="Train smaller, run cheaper">
              <div className="space-y-2">
                <Stat label="3× smaller model" value={fmtN(smallN)} />
                <Stat label="trained on" value={(smallD / 1e9).toFixed(0) + 'B tokens'}
                      sub={(smallD / smallN).toFixed(0) + ' per parameter'} />
                <Stat label="loss penalty" value={'+' + (smallL - best.L).toFixed(3)}
                      tone={smallL - best.L < 0.05 ? 'good' : 'warn'}
                      sub="for 3× cheaper inference, forever" />
              </div>
            </Panel>
          </div>

          <Note kind="warn" title="What this does not tell you">
            <p>These coefficients were fitted on one model family, one dataset and one tokenizer in
            2022. Data quality, architecture and objective all move them. Use it for the shape of
            the tradeoff, not as a number to put in a budget. And loss is not capability — a lower
            loss does not automatically mean a better model at the thing you care about.</p>
          </Note>
        </div>
      </div>
    </Lab>
  )
}
