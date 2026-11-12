import { useMemo, useState } from 'react'
import { rng } from '../lib/tensor'
import { Lab, Panel, Select, Slider, Stat, Toggle } from '../components/ui'

const ROWS = 32, COLS = 128   // a small weight matrix: 32 output channels

type Scheme = 'per-tensor' | 'per-channel' | 'per-group (64)'

/** A weight matrix shaped like a real one: mostly small values, a few channels with outliers. */
function makeWeights(outlierStrength: number) {
  const u = rng(77)
  const w: number[][] = []
  for (let i = 0; i < ROWS; i++) {
    const row: number[] = []
    const isOutlierChannel = i % 11 === 3
    for (let j = 0; j < COLS; j++) {
      const a = Math.max(u(), 1e-12), b = u()
      let v = Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b) * 0.08
      if (isOutlierChannel && j % 17 === 5) v *= outlierStrength
      row.push(v)
    }
    w.push(row)
  }
  return w
}

function quantizeRow(row: number[], bits: number, groupSize: number) {
  const levels = 2 ** (bits - 1) - 1
  const out = new Array(row.length).fill(0)
  for (let g = 0; g < row.length; g += groupSize) {
    const slice = row.slice(g, g + groupSize)
    const scale = Math.max(...slice.map(Math.abs)) / levels || 1e-12
    for (let i = 0; i < slice.length; i++) {
      const q = Math.max(-levels - 1, Math.min(levels, Math.round(slice[i] / scale)))
      out[g + i] = q * scale
    }
  }
  return out
}

export default function QuantizationLab() {
  const [bits, setBits] = useState(4)
  const [scheme, setScheme] = useState<Scheme>('per-channel')
  const [outlier, setOutlier] = useState(12)
  const [showOutliers, setShowOutliers] = useState(true)

  const { w, q, errPerRow, maxErr, meanErr, bytesOrig, bytesQuant } = useMemo(() => {
    const w = makeWeights(showOutliers ? outlier : 1)
    let q: number[][]
    if (scheme === 'per-tensor') {
      const flat = w.flat()
      const levels = 2 ** (bits - 1) - 1
      const scale = Math.max(...flat.map(Math.abs)) / levels || 1e-12
      q = w.map(r => r.map(v => Math.max(-levels - 1, Math.min(levels, Math.round(v / scale))) * scale))
    } else {
      const g = scheme === 'per-channel' ? COLS : 64
      q = w.map(r => quantizeRow(r, bits, g))
    }
    const errPerRow = w.map((r, i) => Math.sqrt(r.reduce((s, v, j) => s + (v - q[i][j]) ** 2, 0) / COLS))
    const all = w.flatMap((r, i) => r.map((v, j) => Math.abs(v - q[i][j])))
    const scalesCount = scheme === 'per-tensor' ? 1 : scheme === 'per-channel' ? ROWS : ROWS * (COLS / 64)
    return {
      w, q, errPerRow,
      maxErr: Math.max(...all),
      meanErr: all.reduce((a, b) => a + b, 0) / all.length,
      bytesOrig: ROWS * COLS * 2,
      bytesQuant: (ROWS * COLS * bits) / 8 + scalesCount * 2,
    }
  }, [bits, scheme, outlier, showOutliers])

  const worstRow = errPerRow.indexOf(Math.max(...errPerRow))
  const bestRow = errPerRow.indexOf(Math.min(...errPerRow))

  // histogram of the weights, with the quantisation grid for the worst row drawn on top
  const hist = useMemo(() => {
    const bins = new Array(60).fill(0)
    const lim = Math.max(...w.flat().map(Math.abs))
    for (const v of w.flat()) {
      const b = Math.min(59, Math.floor(((v + lim) / (2 * lim)) * 60))
      bins[b]++
    }
    return { bins, lim }
  }, [w])

  return (
    <Lab title="Quantisation: putting a grid on real weights"
      takeaway={<>Quantisation error is not about how many bits you have. It is about how far apart
        the biggest and smallest weight in one scaling group are. A single outlier stretches the
        scale and every ordinary weight in that group lands on a coarse grid. Switch to per-tensor
        with outliers on and watch the error explode — that is exactly the problem SmoothQuant and
        AWQ were built to solve.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Slider label="bits" min={2} max={8} value={bits} onChange={setBits} />
              <Select label="one scale per…" value={scheme}
                      options={['per-tensor', 'per-channel', 'per-group (64)'] as const} onChange={setScheme} />
              <Toggle label="outlier channels" value={showOutliers} onChange={setShowOutliers}
                      hint="real models have them" />
              <Slider label="outlier size" min={2} max={40} value={outlier} onChange={setOutlier}
                      fmt={v => `${v}×`} />
            </div>
          </Panel>
          <Stat label="size" value={`${(bytesQuant / 1024).toFixed(1)} KB`}
                sub={`was ${(bytesOrig / 1024).toFixed(1)} KB — ${(bytesOrig / bytesQuant).toFixed(1)}× smaller`} tone="good" />
          <Stat label="average error" value={meanErr.toExponential(2)}
                tone={meanErr > 0.01 ? 'bad' : meanErr > 0.003 ? 'warn' : 'good'} />
          <Stat label="worst error" value={maxErr.toExponential(2)} />
        </div>

        <div className="space-y-4">
          <Panel title="Error per output channel">
            <div className="flex items-end gap-[2px]" style={{ height: 120 }}>
              {errPerRow.map((e, i) => (
                <div key={i} className="group relative flex-1">
                  <div className="absolute bottom-0 w-full rounded-t-[2px]"
                       style={{ height: `${(e / Math.max(...errPerRow)) * 100}%`,
                                background: i === worstRow ? 'var(--bad)' : e > meanErr * 2 ? 'var(--warn)' : 'var(--accent)' }} />
                  <div className="pointer-events-none absolute bottom-full left-1/2 mb-1 hidden -translate-x-1/2 whitespace-nowrap rounded bg-[var(--text)] px-1.5 py-0.5 font-mono text-[10px] text-[#fffdf7] group-hover:block">
                    ch {i}: {e.toExponential(1)}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-2 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Channel {worstRow} is {(errPerRow[worstRow] / errPerRow[bestRow]).toFixed(0)}× worse than
              channel {bestRow}. {scheme === 'per-tensor'
                ? 'With one scale for the whole tensor, one bad channel ruins all of them.'
                : 'Smaller scaling groups keep the damage local.'}
            </div>
          </Panel>

          <Panel title="Weight distribution">
            <div className="flex items-end gap-[2px]" style={{ height: 100 }}>
              {hist.bins.map((n, i) => (
                <div key={i} className="flex-1 rounded-t-[1px] bg-[var(--accent-2)]"
                     style={{ height: `${(n / Math.max(...hist.bins)) * 100}%`, opacity: 0.8 }} />
              ))}
            </div>
            <div className="mt-1 flex justify-between font-mono text-[10px] text-[var(--dim)]">
              <span>{(-hist.lim).toFixed(2)}</span><span>0</span><span>{hist.lim.toFixed(2)}</span>
            </div>
            <div className="mt-2.5 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Almost every weight sits in a narrow band near zero. The grid has
              {' '}{2 ** bits} levels spread across the full range, so if the range is set by a far-off
              outlier, the levels that actually matter are wasted on empty space.
            </div>
          </Panel>

          <Panel title="A single row, before and after">
            <div className="overflow-x-auto">
              <table className="font-mono text-[11px]">
                <tbody>
                  <tr><td className="pr-3 text-[var(--dim)]">original</td>
                    {w[worstRow].slice(0, 14).map((v, i) => <td key={i} className="px-1.5 py-1 text-[#3e392f]">{v.toFixed(3)}</td>)}</tr>
                  <tr><td className="pr-3 text-[var(--dim)]">quantised</td>
                    {q[worstRow].slice(0, 14).map((v, i) => (
                      <td key={i} className={`px-1.5 py-1 ${Math.abs(v - w[worstRow][i]) > meanErr * 3 ? 'text-[var(--bad)]' : 'text-[var(--accent)]'}`}>
                        {v.toFixed(3)}
                      </td>))}</tr>
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
