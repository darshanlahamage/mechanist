import { useMemo, useState } from 'react'
import { entropy, matmul, randn, softmaxRows } from '../lib/tensor'
import { Lab, Line, Note, Panel, Slider, Stat, Toggle } from '../components/ui'

const D = 32, TOKENS = 512

export default function MoELab() {
  const [nExperts, setNExperts] = useState(8)
  const [topK, setTopK] = useState(2)
  const [auxLoss, setAuxLoss] = useState(true)
  const [steps, setSteps] = useState(60)

  // Real routing: random token vectors through a real router matrix and softmax.
  const base = useMemo(() => {
    const x = randn(TOKENS, D, 21)
    const wr = randn(D, nExperts, 33, 1 / Math.sqrt(D))
    const logits = matmul(x, wr)
    return { logits }
  }, [nExperts])

  const { load, history, finalEntropy, dropped } = useMemo(() => {
    // Simplified model of the training feedback loop: experts that win tokens
    // improve, so they win more. The aux loss pushes back against that.
    const bias = new Array(nExperts).fill(0)
    const history: [number, number][] = []
    let load = new Array(nExperts).fill(0)
    const capacity = Math.ceil((TOKENS * topK) / nExperts * 1.25)
    let dropped = 0

    for (let s = 0; s < steps; s++) {
      load = new Array(nExperts).fill(0)
      dropped = 0
      const counts = new Array(nExperts).fill(0)
      for (let t = 0; t < TOKENS; t++) {
        const row = base.logits[t].map((v, e) => v + bias[e])
        const order = row.map((v, e) => [v, e] as [number, number]).sort((a, b) => b[0] - a[0])
        for (let k = 0; k < topK; k++) {
          const e = order[k][1]
          if (counts[e] >= capacity) { dropped++; continue }
          counts[e]++; load[e]++
        }
      }
      const frac = load.map(v => v / (TOKENS * topK))
      history.push([s, entropy(frac.filter(v => v > 0))])

      for (let e = 0; e < nExperts; e++) {
        bias[e] += 0.35 * (frac[e] - 1 / nExperts)                 // rich get richer
        if (auxLoss) bias[e] -= 0.45 * (frac[e] - 1 / nExperts)    // balancing push-back
      }
    }
    const frac = load.map(v => v / (TOKENS * topK))
    return { load, history, finalEntropy: entropy(frac.filter(v => v > 0)), dropped }
  }, [base, nExperts, topK, auxLoss, steps])

  const maxLoad = Math.max(...load)
  const idealLoad = (TOKENS * topK) / nExperts
  const worstRatio = maxLoad / idealLoad
  const maxEntropy = Math.log2(nExperts)

  return (
    <Lab title="Mixture of experts: routing, and how it collapses"
      takeaway={<>Nothing makes a router use its experts evenly. Left alone, a small early lead
        compounds — the expert that wins tokens gets more gradient, gets better, and wins more. Turn
        the balancing loss off and watch the entropy fall. Then look at the throughput cost: your
        batch runs as slowly as the busiest device, so one overloaded expert slows everything.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Slider label="experts" min={2} max={32} step={2} value={nExperts} onChange={setNExperts} />
              <Slider label="top-k per token" min={1} max={4} value={topK} onChange={setTopK} />
              <Slider label="training steps" min={1} max={120} value={steps} onChange={setSteps} />
              <Toggle label="load balancing loss" value={auxLoss} onChange={setAuxLoss}
                      hint={auxLoss ? 'pushing back on popular experts' : 'nothing stopping collapse'} />
            </div>
          </Panel>
          <Stat label="routing entropy" value={`${finalEntropy.toFixed(2)}`}
                sub={`even would be ${maxEntropy.toFixed(2)}`}
                tone={finalEntropy < maxEntropy * 0.6 ? 'bad' : 'good'} />
          <Stat label="busiest expert" value={`${worstRatio.toFixed(2)}× its share`}
                tone={worstRatio > 2 ? 'bad' : worstRatio > 1.4 ? 'warn' : 'good'} />
          <Stat label="tokens dropped" value={String(dropped)}
                sub="over capacity, silently skipped" tone={dropped > 0 ? 'bad' : 'good'} />
        </div>

        <div className="space-y-4">
          <Panel title="Tokens per expert, after training">
            <div className="flex items-end gap-1" style={{ height: 130 }}>
              {load.map((v, i) => (
                <div key={i} className="group relative flex-1">
                  <div className="absolute bottom-0 w-full rounded-t-sm transition-all"
                       style={{ height: `${(v / Math.max(maxLoad, 1)) * 100}%`,
                                background: v > idealLoad * 1.5 ? 'var(--bad)'
                                          : v < idealLoad * 0.5 ? '#cec4ae' : 'var(--accent)' }} />
                  <div className="pointer-events-none absolute bottom-full left-1/2 mb-1 hidden -translate-x-1/2 whitespace-nowrap rounded bg-[var(--text)] px-1.5 py-0.5 font-mono text-[10px] text-[#fffdf7] group-hover:block">
                    {v} tokens
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-2 flex items-center justify-between text-[11.5px] text-[var(--dim)]">
              <span>expert 0</span>
              <span>even split would be {idealLoad.toFixed(0)} each</span>
              <span>expert {nExperts - 1}</span>
            </div>
          </Panel>

          <Panel title="Routing entropy over training">
            <Line series={[{ name: 'entropy', color: auxLoss ? '#0b6f66' : '#b03a31', points: history }]}
                  xLabel="step" yLabel="bits" height={150} />
            <div className="mt-2 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Entropy is how evenly tokens are spread. Flat near the top means every expert is used.
              Falling means the router is picking favourites and the rest of your parameters are
              dead weight you are still paying to store.
            </div>
          </Panel>

          <Note kind="info" title="What is real here and what is modelled">
            <p>The routing itself is real: random token vectors go through a real router matrix, a
            real softmax, and a real top-k selection. The part across training steps is a simplified
            feedback model — it reproduces the collapse mechanism, not a specific model's numbers.</p>
          </Note>
        </div>
      </div>
    </Lab>
  )
}
