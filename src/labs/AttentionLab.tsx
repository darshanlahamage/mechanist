import { useMemo, useState } from 'react'
import { attentionHead, randn, entropy } from '../lib/tensor'
import { Bars, Heat, Lab, Panel, Select, Slider, Stat, Toggle } from '../components/ui'

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat']
const D_MODEL = 16

export default function AttentionLab() {
  const [dHead, setDHead] = useState(16)
  const [scaled, setScaled] = useState(true)
  const [causal, setCausal] = useState(true)
  const [query, setQuery] = useState(5)
  const [view, setView] = useState<'weights' | 'scores' | 'masked'>('weights')

  const r = useMemo(() => {
    const x = randn(TOKENS.length, D_MODEL, 7)
    const wq = randn(D_MODEL, dHead, 11, 1 / Math.sqrt(D_MODEL))
    const wk = randn(D_MODEL, dHead, 13, 1 / Math.sqrt(D_MODEL))
    const wv = randn(D_MODEL, dHead, 17, 1 / Math.sqrt(D_MODEL))
    return attentionHead(x, wq, wk, wv, { causal, scaled })
  }, [dHead, scaled, causal])

  const row = r.weights[query]
  const h = entropy(row)
  const maxH = Math.log2(causal ? query + 1 : TOKENS.length)
  const peak = Math.max(...row)
  const shown = view === 'weights' ? r.weights : view === 'scores' ? r.scores : r.masked

  return (
    <Lab title="Attention, one head, every step"
      takeaway={<>Turn off <b>divide by √d</b> and watch the attention collapse onto a single token.
        That is the whole reason the <code>1/√d</code> is in the formula — without it the scores grow with
        head size, softmax saturates, and gradients die.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Slider label="head size (d_head)" min={2} max={64} step={2} value={dHead} onChange={setDHead} />
              <Toggle label="divide by √d" value={scaled} onChange={setScaled}
                      hint={scaled ? `scores ÷ ${Math.sqrt(dHead).toFixed(2)}` : 'raw dot products'} />
              <Toggle label="causal mask" value={causal} onChange={setCausal}
                      hint="can only look left" />
              <Slider label="which token is asking" min={0} max={TOKENS.length - 1} value={query}
                      onChange={setQuery} fmt={v => `"${TOKENS[v]}"`} />
              <Select label="show matrix" value={view} options={['weights', 'scores', 'masked'] as const}
                      onChange={setView} />
            </div>
          </Panel>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="peak weight" value={peak.toFixed(3)} tone={peak > 0.9 ? 'bad' : 'good'} />
            <Stat label="entropy" value={`${h.toFixed(2)}`} sub={`max ${maxH.toFixed(2)} bits`}
                  tone={h < maxH * 0.35 ? 'bad' : 'normal'} />
          </div>
          {!scaled && dHead >= 32 && (
            <div className="rounded-lg border-l-2 border-[var(--bad)] bg-[rgba(248,113,113,0.07)] px-3 py-2 text-[12px] text-[#c7cdd8]">
              Scores now have standard deviation ≈ {Math.sqrt(dHead).toFixed(1)}. Softmax turned into
              a hard pick. In training, gradients through this are near zero.
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Panel title={
            view === 'weights' ? 'Attention weights — each row sums to 1'
            : view === 'scores' ? 'Q·Kᵀ scores' + (scaled ? ' ÷ √d' : ' (raw)')
            : 'After the causal mask (dark = blocked)'}>
            <Heat m={shown} rowLabels={TOKENS} colLabels={TOKENS} cell={38} showValues />
          </Panel>

          <Panel title={`What "${TOKENS[query]}" actually looks at`}>
            <Bars values={row} labels={TOKENS} highlight={row.indexOf(peak)} />
            <div className="mt-3 text-[12.5px] leading-relaxed text-[var(--dim)]">
              The output for this token is the sum of every value vector, weighted by these bars.
              {causal && query < TOKENS.length - 1 && <> Positions to the right are zero because the mask set them to −∞ before the softmax.</>}
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
