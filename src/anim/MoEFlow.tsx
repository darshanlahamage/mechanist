import { useMemo } from 'react'
import { matmul, randn, softmaxRows } from '../lib/tensor'
import { Scene, appear, ease, stagger, type Frame, type Step } from './Scene'

const W = 780, H = 400
const N_EXPERTS = 8, N_TOK = 24, D = 24, TOP_K = 2

const STEPS: Step[] = [
  { label: 'A batch of tokens', say: 'A dense model would push every one of these through the same feed-forward block, using every weight it has.', hold: 2.4 },
  { label: 'A tiny router scores the experts', say: 'One small matrix turns each token into eight numbers — how well it matches each expert. Softmax, then take the best two.', hold: 3.6 },
  { label: 'Each token goes to its two experts', say: 'Only those two run. The other six never see this token, so you store eight blocks but pay for two.', hold: 5 },
  { label: 'A healthy spread', say: 'With the balancing loss on, the work lands roughly evenly. Every expert is busy, every parameter you are paying to store is doing something.', hold: 3.6 },
  { label: 'Take the balancing loss away', say: 'An expert that wins tokens gets more gradient, gets better, and wins more. The lead compounds.', hold: 5 },
  { label: 'Collapse', say: 'Two experts take almost everything. The rest are dead weight you still hold in memory — and the batch now runs at the speed of the busiest device, with overflow tokens silently skipping the layer.', hold: 4.5 },
]

export default function MoEFlow() {
  const { picks, collapsedPicks, balanced, collapsed } = useMemo(() => {
    const x = randn(N_TOK, D, 21)
    const wr = randn(D, N_EXPERTS, 41, 1 / Math.sqrt(D))
    const probs = softmaxRows(matmul(x, wr))
    const picks = probs.map(row =>
      row.map((v, e) => [v, e] as [number, number])
         .sort((a, b) => b[0] - a[0]).slice(0, TOP_K))

    const balanced = new Array(N_EXPERTS).fill(0)
    picks.forEach(p => p.forEach(([, e]) => balanced[e]++))

    // what the same router does once a couple of experts have run away with it
    // a lead that has already compounded: added to the router's probabilities
    const bias = [0.25, 0.1125, 0.175, 0.05, 0.0125, 0, -0.0125, -0.025]
    const collapsedPicks = probs.map(row =>
      row.map((v, e) => [v + bias[e], e] as [number, number])
         .sort((a, b) => b[0] - a[0]).slice(0, TOP_K))
    const collapsed = new Array(N_EXPERTS).fill(0)
    collapsedPicks.forEach(p => p.forEach(([, e]) => collapsed[e]++))
    return { picks, collapsedPicks, balanced, collapsed }
  }, [])

  const tokX = (i: number) => 46 + (i % 12) * 61
  const tokY = (i: number) => (i < 12 ? 40 : 66)
  const expX = (e: number) => 66 + e * 93
  const expY = 300

  const draw = ({ step, t }: Frame) => {
    const e = ease(t)
    const load = step >= 4 ? collapsed : balanced
    const cap = Math.ceil((N_TOK * TOP_K) / N_EXPERTS * 1.25)
    const maxLoad = Math.max(...load, 1)

    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: 'block' }}>
        {/* tokens */}
        {Array.from({ length: N_TOK }, (_, i) => (
          <rect key={i} x={tokX(i) - 11} y={tokY(i) - 8} width={22} height={16} rx={3}
            fill="#fffdf7" stroke="#dcd3c0"
            opacity={step === 0 ? 0.25 + 0.75 * stagger(e, i, N_TOK, 0.93) : 1} />
        ))}
        <text x={W / 2} y={18} textAnchor="middle" fontSize="11" fill="#8a8171">24 tokens</text>

        {/* router */}
        {step >= 1 && (
          <g opacity={step === 1 ? appear(e, 0, 0.25) : 1}>
            <rect x={W / 2 - 90} y={110} width={180} height={30} rx={6}
              fill="rgba(106,70,163,0.12)" stroke="#6a46a3" />
            <text x={W / 2} y={130} textAnchor="middle" fontSize="12" fill="#6a46a3">router — one small matrix</text>
            {Array.from({ length: N_TOK }, (_, i) => (
              <line key={i} x1={tokX(i)} y1={tokY(i) + 9} x2={W / 2} y2={110}
                stroke="#cec4ae" strokeWidth="0.6" opacity={0.5 * appear(e, 0, 0.3)} />
            ))}
          </g>
        )}

        {/* experts */}
        {step >= 1 && Array.from({ length: N_EXPERTS }, (_, ex) => {
          const over = load[ex] > cap
          const busy = load[ex] / maxLoad
          const dead = step >= 5 && load[ex] <= 1
          return (
            <g key={ex}>
              <rect x={expX(ex) - 38} y={expY - 18} width={76} height={36} rx={6}
                fill={dead ? '#f0eadc' : `rgba(11,111,102,${0.06 + busy * 0.2})`}
                stroke={over ? '#b03a31' : dead ? '#dcd3c0' : '#0b6f66'}
                strokeOpacity={dead ? 1 : 0.75} />
              <text x={expX(ex)} y={expY + 5} textAnchor="middle" fontSize="11"
                fill={dead ? '#a79e8c' : '#3e392f'}>E{ex}</text>
              {step >= 3 && (
                <>
                  <rect x={expX(ex) - 38} y={expY + 28} width={76} height={7} rx={3} fill="#fffdf7" />
                  <rect x={expX(ex) - 38} y={expY + 28} rx={3} height={7}
                    width={76 * Math.min(1, load[ex] / (cap * 1.3)) * (step === 3 || step === 5 ? e : 1)}
                    fill={over ? '#b03a31' : '#0b6f66'} />
                  <text x={expX(ex)} y={expY + 50} textAnchor="middle" fontSize="10"
                    fontFamily="ui-monospace" fill={over ? '#b03a31' : dead ? '#a79e8c' : '#7b7364'}>
                    {load[ex]}{over ? ` (${load[ex] - cap} dropped)` : ''}
                  </text>
                </>
              )}
            </g>
          )
        })}

        {/* tokens travelling to their experts */}
        {step >= 2 && Array.from({ length: N_TOK }, (_, i) => {
          // once the router has collapsed, the same tokens go somewhere else
          const dests = (step >= 4 ? collapsedPicks[i] : picks[i]).map(c => c[1])

          const arrive = step === 2 ? stagger(e, i, N_TOK, 0.82)
            : step === 4 ? stagger(e, i, N_TOK, 0.86) : 1
          if (arrive <= 0) return null

          return dests.map((ex, k) => {
            const sx = W / 2, sy = 140
            const dx = expX(ex) + (k ? 12 : -12), dy = expY - 18
            const px = sx + (dx - sx) * arrive
            const py = sy + (dy - sy) * arrive
            return (
              <g key={`${i}-${k}`}>
                <line x1={sx} y1={sy} x2={px} y2={py} stroke="#0b6f66"
                  strokeWidth="0.7" strokeOpacity={0.14} />
                <circle cx={px} cy={py} r="3" fill="#0b6f66" opacity={0.55 + 0.45 * arrive} />
              </g>
            )
          })
        })}

        {step >= 3 && (
          <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="10.5"
            fill={step >= 5 ? '#b03a31' : '#8a8171'}>
            {step >= 5
              ? 'capacity exceeded on the busy experts — those tokens skip the layer entirely'
              : `capacity per expert: ${cap} tokens`}
          </text>
        )}
      </svg>
    )
  }

  return <Scene steps={STEPS} height={400} autoPlay>{draw}</Scene>
}
