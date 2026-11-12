import { useMemo, useState } from 'react'
import { randn, type Mat } from '../lib/tensor'
import { Heat, Lab, Panel, Select, Slider, Stat } from '../components/ui'

const B = 8, F = 16   // 8 tokens in the batch, 16 features each
type Norm = 'none' | 'LayerNorm' | 'RMSNorm' | 'BatchNorm'

export default function NormLab() {
  const [norm, setNorm] = useState<Norm>('LayerNorm')
  const [drift, setDrift] = useState(3)
  const [outlierFeature, setOutlierFeature] = useState(6)

  const x = useMemo(() => {
    const base = randn(B, F, 9)
    return base.map((row, i) => row.map((v, j) => {
      let out = v + (i - B / 2) * drift * 0.4          // each token drifts to its own scale
      if (j === outlierFeature) out *= 4               // one feature is much larger
      return out
    }))
  }, [drift, outlierFeature])

  const y: Mat = useMemo(() => {
    const eps = 1e-5
    if (norm === 'none') return x
    if (norm === 'LayerNorm') {
      return x.map(row => {
        const m = row.reduce((a, b) => a + b, 0) / F
        const v = row.reduce((a, b) => a + (b - m) ** 2, 0) / F
        return row.map(e => (e - m) / Math.sqrt(v + eps))
      })
    }
    if (norm === 'RMSNorm') {
      return x.map(row => {
        const ms = row.reduce((a, b) => a + b * b, 0) / F
        return row.map(e => e / Math.sqrt(ms + eps))
      })
    }
    // BatchNorm: statistics down each column, across the batch
    const out = x.map(r => [...r])
    for (let j = 0; j < F; j++) {
      const col = x.map(r => r[j])
      const m = col.reduce((a, b) => a + b, 0) / B
      const v = col.reduce((a, b) => a + (b - m) ** 2, 0) / B
      for (let i = 0; i < B; i++) out[i][j] = (x[i][j] - m) / Math.sqrt(v + eps)
    }
    return out
  }, [x, norm])

  const rowStats = y.map(r => {
    const m = r.reduce((a, b) => a + b, 0) / F
    return { mean: m, std: Math.sqrt(r.reduce((a, b) => a + (b - m) ** 2, 0) / F) }
  })
  const colStats = Array.from({ length: F }, (_, j) => {
    const col = y.map(r => r[j])
    const m = col.reduce((a, b) => a + b, 0) / B
    return { mean: m, std: Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / B) }
  })

  const rowSpread = Math.max(...rowStats.map(s => s.std)) / Math.min(...rowStats.map(s => s.std))
  const explain: Record<Norm, string> = {
    'none': 'Tokens are on completely different scales. Whatever comes next has to cope with all of them at once.',
    'LayerNorm': 'Each row is centred and scaled on its own. Every token now looks the same size, and the batch is irrelevant.',
    'RMSNorm': 'Same as LayerNorm but it skips the mean subtraction. Cheaper, one less pass over the data, and in practice just as good — which is why Llama and most modern models use it.',
    'BatchNorm': 'Statistics are taken down each column, across the batch. That means the result for one token depends on the other tokens in the batch, which is why this breaks for language models and for batch size 1.',
  }

  return (
    <Lab title="What each normaliser actually normalises"
      takeaway={<>The only real difference is which direction you take the average over. LayerNorm
        and RMSNorm go across the features of one token, so each token is handled independently.
        BatchNorm goes across the batch, so a token's output depends on which other tokens happened
        to be in the batch with it. For variable-length text and batch size 1 at inference, that is
        fatal — which is why transformers never used it.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Select label="normaliser" value={norm}
                      options={['none', 'LayerNorm', 'RMSNorm', 'BatchNorm'] as const} onChange={setNorm} />
              <Slider label="scale drift between tokens" min={0} max={8} step={0.5} value={drift}
                      onChange={setDrift} fmt={v => v.toFixed(1)} />
              <Slider label="which feature is an outlier" min={0} max={F - 1} value={outlierFeature}
                      onChange={setOutlierFeature} />
            </div>
          </Panel>
          <Stat label="biggest / smallest row spread" value={`${rowSpread.toFixed(2)}×`}
                tone={rowSpread > 2 ? 'bad' : rowSpread > 1.2 ? 'warn' : 'good'}
                sub="1.0 means every token is on the same scale" />
          <div className="rounded-lg border border-[var(--line)] bg-[var(--panel-2)] px-3 py-2.5 text-[12px] leading-relaxed text-[var(--dim)]">
            {explain[norm]}
          </div>
        </div>

        <div className="space-y-4">
          <Panel title="Before and after — rows are tokens, columns are features">
            <div className="flex flex-wrap gap-6">
              <Heat m={x} cell={17} title="input" />
              <Heat m={y} cell={17} title={norm === 'none' ? 'unchanged' : `after ${norm}`} />
            </div>
          </Panel>

          <div className="grid gap-3 lg:grid-cols-2">
            <Panel title="Per token (across a row)">
              <table className="w-full font-mono text-[11.5px]">
                <thead><tr className="text-[var(--dim)]">
                  <th className="py-1 text-left font-normal">token</th>
                  <th className="py-1 text-right font-normal">mean</th>
                  <th className="py-1 text-right font-normal">std</th>
                </tr></thead>
                <tbody>{rowStats.map((s, i) => (
                  <tr key={i} className="border-t border-[var(--line)]">
                    <td className="py-1 text-[#3e392f]">{i}</td>
                    <td className="py-1 text-right text-[var(--dim)]">{s.mean.toFixed(3)}</td>
                    <td className="py-1 text-right text-[var(--accent)]">{s.std.toFixed(3)}</td>
                  </tr>))}</tbody>
              </table>
            </Panel>
            <Panel title="Per feature (down a column)">
              <table className="w-full font-mono text-[11.5px]">
                <thead><tr className="text-[var(--dim)]">
                  <th className="py-1 text-left font-normal">feature</th>
                  <th className="py-1 text-right font-normal">mean</th>
                  <th className="py-1 text-right font-normal">std</th>
                </tr></thead>
                <tbody>{colStats.slice(0, 8).map((s, j) => (
                  <tr key={j} className="border-t border-[var(--line)]">
                    <td className={`py-1 ${j === outlierFeature ? 'text-[var(--warn)]' : 'text-[#3e392f]'}`}>
                      {j}{j === outlierFeature && ' (outlier)'}
                    </td>
                    <td className="py-1 text-right text-[var(--dim)]">{s.mean.toFixed(3)}</td>
                    <td className="py-1 text-right text-[var(--accent)]">{s.std.toFixed(3)}</td>
                  </tr>))}</tbody>
              </table>
            </Panel>
          </div>
          <div className="text-[12.5px] leading-relaxed text-[var(--dim)]">
            Watch which table goes to mean 0 and std 1. LayerNorm and RMSNorm fix the left one.
            BatchNorm fixes the right one. They are solving different problems.
          </div>
        </div>
      </div>
    </Lab>
  )
}
