import { useMemo, useState } from 'react'
import { entropy, rng, softmaxRows } from '../lib/tensor'
import { Lab, Panel, Slider, Stat, Toggle } from '../components/ui'

const CANDIDATES = ['the', 'a', 'my', 'this', 'that', 'one', 'some', 'his',
                    'their', 'its', 'our', 'your', 'another', 'each', 'every', 'no']
const BASE_LOGITS = [4.2, 3.8, 2.9, 2.6, 2.4, 1.8, 1.5, 1.2, 1.0, 0.7, 0.3, 0.0, -0.4, -0.9, -1.5, -2.2]

export default function SamplingLab() {
  const [temp, setTemp] = useState(1.0)
  const [topK, setTopK] = useState(16)
  const [topP, setTopP] = useState(1.0)
  const [minP, setMinP] = useState(0)
  const [greedy, setGreedy] = useState(false)

  const { probs, kept, samples, h } = useMemo(() => {
    const t = greedy ? 0.01 : Math.max(temp, 0.01)
    const scaled = BASE_LOGITS.map(v => v / t)
    let p = softmaxRows([scaled])[0]

    const order = p.map((v, i) => [v, i] as [number, number]).sort((a, b) => b[0] - a[0])
    const kept = new Set<number>()

    // top-k
    const kLimit = greedy ? 1 : topK
    order.slice(0, kLimit).forEach(([, i]) => kept.add(i))
    // top-p (nucleus): keep the smallest set whose mass reaches p
    if (topP < 1 && !greedy) {
      const nucleus = new Set<number>()
      let cum = 0
      for (const [v, i] of order) { nucleus.add(i); cum += v; if (cum >= topP) break }
      for (const i of [...kept]) if (!nucleus.has(i)) kept.delete(i)
    }
    // min-p: drop anything below a fraction of the top probability
    if (minP > 0 && !greedy) {
      const floor = order[0][0] * minP
      for (const i of [...kept]) if (p[i] < floor) kept.delete(i)
    }

    const total = [...kept].reduce((s, i) => s + p[i], 0)
    const final = p.map((v, i) => (kept.has(i) ? v / total : 0))

    const u = rng(2024)
    const samples: string[] = []
    for (let s = 0; s < 24; s++) {
      let r = u(), acc = 0, pick = 0
      for (let i = 0; i < final.length; i++) { acc += final[i]; if (r <= acc) { pick = i; break } }
      samples.push(CANDIDATES[pick])
    }
    return { probs: final, kept, samples, h: entropy(final.filter(v => v > 0)) }
  }, [temp, topK, topP, minP, greedy])

  const unique = new Set(samples).size

  return (
    <Lab title="Same logits, every decoding strategy"
      takeaway={<>Temperature reshapes the whole distribution. Top-k and top-p only cut the tail off.
        They do different jobs, which is why people stack them. Set temperature to 2 with no cutoff
        and you will see the model pick words it assigned almost no probability to — that is where
        incoherent text comes from, not from the model being wrong.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Toggle label="greedy" value={greedy} onChange={setGreedy} hint="always take the top token" />
              <Slider label="temperature" min={0.1} max={2.5} step={0.05} value={temp}
                      onChange={setTemp} fmt={v => v.toFixed(2)} />
              <Slider label="top-k" min={1} max={16} value={topK} onChange={setTopK} />
              <Slider label="top-p" min={0.1} max={1} step={0.01} value={topP}
                      onChange={setTopP} fmt={v => v.toFixed(2)} />
              <Slider label="min-p" min={0} max={0.5} step={0.01} value={minP}
                      onChange={setMinP} fmt={v => v.toFixed(2)} />
            </div>
          </Panel>
          <Stat label="tokens in play" value={String(kept.size)} sub={`of ${CANDIDATES.length}`}
                tone={kept.size === 1 ? 'warn' : 'good'} />
          <Stat label="entropy" value={`${h.toFixed(2)} bits`}
                sub={h < 0.5 ? 'almost deterministic' : h > 3 ? 'very random' : 'balanced'} />
        </div>

        <div className="space-y-4">
          <Panel title="Probability of each next token">
            <div className="space-y-1">
              {CANDIDATES.map((c, i) => (
                <div key={c} className="flex items-center gap-2">
                  <span className="w-[62px] shrink-0 text-right font-mono text-[11px] text-[var(--dim)]">{c}</span>
                  <div className="h-[14px] flex-1 overflow-hidden rounded-sm bg-[var(--panel-2)]">
                    <div className="h-full rounded-sm transition-all"
                         style={{ width: `${probs[i] * 100}%`,
                                  background: kept.has(i) ? 'var(--accent)' : '#cec4ae' }} />
                  </div>
                  <span className={`w-[52px] shrink-0 font-mono text-[11px] ${probs[i] > 0 ? 'text-[#3e392f]' : 'text-[#a79e8c]'}`}>
                    {probs[i] > 0 ? (probs[i] * 100).toFixed(1) + '%' : 'cut'}
                  </span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="24 samples drawn from that distribution">
            <div className="flex flex-wrap gap-1.5">
              {samples.map((s, i) => (
                <span key={i} className="rounded bg-[#e3dbc9] px-2 py-0.5 font-mono text-[11.5px] text-[#3e392f]">{s}</span>
              ))}
            </div>
            <div className="mt-2.5 text-[12px] text-[var(--dim)]">
              {unique} different words out of 24 draws.
              {unique === 1 && ' Completely deterministic — this is what greedy decoding gives you.'}
              {unique > 9 && ' Very scattered. Text like this reads as rambling.'}
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
