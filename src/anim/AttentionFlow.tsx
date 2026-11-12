import { useMemo } from 'react'
import { attentionHead, randn } from '../lib/tensor'
import { Scene, appear, ease, stagger, type Frame, type Step } from './Scene'

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat']
const Q_IDX = 5          // the token doing the asking
const D_MODEL = 16, D_HEAD = 8
const W = 780, H = 430

const STEPS: Step[] = [
  { label: 'Six tokens arrive', say: 'The sentence is already numbers — each token is a vector. On their own they know nothing about each other.', hold: 2.4 },
  { label: 'Each token makes three vectors', say: 'Three learned matrices turn every token into a query (what am I looking for), a key (what do I contain) and a value (what I will hand over).', hold: 3.4 },
  { label: '"mat" asks a question', say: 'Take one token. Its query is a direction in space. Everything that follows is about finding which keys point the same way.', hold: 2.6 },
  { label: 'Compare the query with every key', say: 'One dot product per token. A big number means that key points the same way as the query — they match.', hold: 4 },
  { label: 'Divide by the square root of the head size', say: 'Dot products of 8-dimensional vectors drift about ±2.8. Divide by √8 and they come back to about ±1, so the next step stays sensitive.', hold: 3 },
  { label: 'Block anything to the right', say: 'A token predicting the next word may not read the future. Those scores are set to minus infinity before the softmax, so they become exactly zero.', hold: 2.8 },
  { label: 'Softmax turns scores into shares', say: 'Exponentiate, then divide by the total. Now the numbers are positive and add up to 1 — they are how much of each token to take.', hold: 3.4 },
  { label: 'Mix the values by those shares', say: 'Each value vector flows along its edge, thicker where the weight is larger. They add up, and that sum is the new vector for "mat".', hold: 4 },
  { label: 'Every token does this at once', say: 'The same thing happens for all six queries in one matrix multiply. That is the whole layer — and why it parallelises where a recurrent network could not.', hold: 4 },
]

const colA = '#5eead4', colB = '#a78bfa', colC = '#fbbf24'

function Strip({ v, x, y, w = 76, h = 13, color, op = 1 }:
  { v: number[]; x: number; y: number; w?: number; h?: number; color: string; op?: number }) {
  const n = v.length, cw = w / n
  const max = Math.max(...v.map(Math.abs)) || 1
  return (
    <g opacity={op}>
      {v.map((val, i) => (
        <rect key={i} x={x + i * cw} y={y} width={cw - 0.8} height={h} rx={1}
          fill={color} opacity={0.22 + 0.78 * Math.abs(val) / max} />
      ))}
      <rect x={x} y={y} width={w} height={h} rx={2} fill="none" stroke={color} strokeOpacity={0.35} />
    </g>
  )
}

export default function AttentionFlow() {
  const data = useMemo(() => {
    const x = randn(TOKENS.length, D_MODEL, 7)
    const wq = randn(D_MODEL, D_HEAD, 11, 1 / Math.sqrt(D_MODEL))
    const wk = randn(D_MODEL, D_HEAD, 13, 1 / Math.sqrt(D_MODEL))
    const wv = randn(D_MODEL, D_HEAD, 17, 1 / Math.sqrt(D_MODEL))
    return attentionHead(x, wq, wk, wv, { causal: true, scaled: true })
  }, [])

  const tokX = (i: number) => 58 + i * 118
  const qx = tokX(Q_IDX), qy = 302

  const draw = ({ step, t }: Frame) => {
    const e = ease(t)
    const raw = data.rawScores[Q_IDX]
    const scaled = data.scores[Q_IDX]
    const weights = data.weights[Q_IDX]

    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: 'block' }}>
        <defs>
          <linearGradient id="flow" x1="0" x2="1">
            <stop offset="0%" stopColor={colC} stopOpacity="0.1" />
            <stop offset="100%" stopColor={colC} stopOpacity="0.9" />
          </linearGradient>
        </defs>

        {/* ---- faint outline of where this is going ---- */}
        {step <= 2 && (
          <g opacity={(step === 0 ? 0.5 : step === 1 ? 0.4 : 0.25 * (1 - e)) } stroke="#2b3340"
             strokeDasharray="3 4" fill="none">
            <rect x={60} y={158} width={W - 120} height={62} rx={6} />
            <text x={W / 2} y={194} textAnchor="middle" fontSize="10.5" fill="#39414e"
                  stroke="none" strokeDasharray="0">scores, then shares of attention</text>
            <rect x={qx - 60} y={qy - 16} width={120} height={32} rx={6} />
            <rect x={qx - 70} y={388} width={140} height={30} rx={6} />
            <text x={qx} y={407} textAnchor="middle" fontSize="10.5" fill="#39414e"
                  stroke="none" strokeDasharray="0">the answer</text>
          </g>
        )}

        {/* ---- tokens ---- */}
        {TOKENS.map((tk, i) => {
          const on = step === 0 ? stagger(e, i, 6) : 1
          const dim = step >= 2 && step <= 7 && i > Q_IDX ? 0.25 : 1
          return (
            <g key={i} opacity={on * dim}>
              <rect x={tokX(i) - 38} y={22} width={76} height={26} rx={5}
                fill={i === Q_IDX && step >= 2 ? 'rgba(94,234,212,0.16)' : '#1a1f27'}
                stroke={i === Q_IDX && step >= 2 ? colA : '#2b3340'} />
              <text x={tokX(i)} y={39} textAnchor="middle" fontSize="12.5"
                fill={i === Q_IDX && step >= 2 ? colA : '#c7cdd8'}>{tk}</text>
            </g>
          )
        })}

        {/* ---- Q K V strips ---- */}
        {step >= 1 && TOKENS.map((_, i) => {
          const a = step === 1 ? stagger(e, i, 6) : 1
          const kOn = step >= 3 ? 1 : a
          return (
            <g key={i}>
              <Strip v={data.q[i]} x={tokX(i) - 38} y={60} color={colA}
                op={a * (step >= 2 ? (i === Q_IDX ? 1 : 0.18) : 1)} />
              <Strip v={data.k[i]} x={tokX(i) - 38} y={80} color={colB} op={kOn * (step >= 6 && i > Q_IDX ? 0.2 : 1)} />
              <Strip v={data.v[i]} x={tokX(i) - 38} y={100} color={colC}
                op={step >= 7 ? 1 : a * 0.75} />
              {step === 1 && (
                <g opacity={a} fontSize="8.5" fill="#6b7483">
                  <text x={tokX(i) - 46} y={70} textAnchor="end">Q</text>
                  <text x={tokX(i) - 46} y={90} textAnchor="end">K</text>
                  <text x={tokX(i) - 46} y={110} textAnchor="end">V</text>
                </g>
              )}
            </g>
          )
        })}

        {/* ---- the asking token's query, pulled out ---- */}
        {step >= 2 && (
          <g opacity={step === 2 ? appear(e) : 1}>
            <rect x={qx - 60} y={qy - 16} width={120} height={32} rx={6}
              fill="rgba(94,234,212,0.1)" stroke={colA} />
            <text x={qx} y={qy + 5} textAnchor="middle" fontSize="12" fill={colA}>query of "mat"</text>
            <Strip v={data.q[Q_IDX]} x={qx - 38} y={qy + 22} color={colA} />
          </g>
        )}

        {/* ---- comparison lines + scores ---- */}
        {step >= 3 && TOKENS.map((_, i) => {
          const blocked = i > Q_IDX
          const prog = step === 3 ? stagger(e, i, 6, 0.4) : 1
          if (prog <= 0) return null
          const path = `M ${qx} ${qy - 18} C ${qx} ${qy - 90}, ${tokX(i)} ${200}, ${tokX(i)} ${120}`
          const wt = weights[i]
          const thick = step >= 7 ? 1 + wt * 13 : 1.2
          const show = step >= 5 && blocked ? 0.08 : 0.55 + (step >= 6 ? wt * 0.45 : 0)
          return (
            <g key={i} opacity={prog}>
              <path d={path} fill="none"
                stroke={step >= 7 ? 'url(#flow)' : colB}
                strokeWidth={step === 7 ? 1 + wt * 13 * ease(t) : thick}
                strokeOpacity={show}
                strokeDasharray={step === 3 ? '400' : undefined}
                strokeDashoffset={step === 3 ? 400 * (1 - ease(stagger(e, i, 6, 0.4))) : undefined} />
              {step >= 3 && step <= 6 && (
                <text x={tokX(i)} y={138} textAnchor="middle" fontSize="11" fontFamily="ui-monospace"
                  fill={step >= 5 && blocked ? '#f87171' : '#c7cdd8'} opacity={prog}>
                  {step === 3 ? raw[i].toFixed(2)
                    : step === 4 ? (raw[i] + (scaled[i] - raw[i]) * e).toFixed(2)
                    : step >= 5 && blocked ? '−∞'
                    : scaled[i].toFixed(2)}
                </text>
              )}
            </g>
          )
        })}

        {step === 4 && (
          <text x={W / 2} y={178} textAnchor="middle" fontSize="12" fill={colA} opacity={appear(e)}>
            ÷ √8 = ÷ 2.83
          </text>
        )}

        {/* ---- softmax weights ---- */}
        {step >= 6 && TOKENS.map((_, i) => {
          const wt = weights[i]
          const g = step === 6 ? ease(t) : 1
          const hgt = wt * 60 * g
          return (
            <g key={i}>
              <rect x={tokX(i) - 24} y={200 - hgt} width={48} height={hgt} rx={2}
                fill={colA} opacity={0.85} />
              <text x={tokX(i)} y={216} textAnchor="middle" fontSize="10.5" fontFamily="ui-monospace"
                fill={wt > 0.001 ? colA : '#454d5a'}>{(wt * g).toFixed(3)}</text>
            </g>
          )
        })}
        {step >= 6 && (
          <text x={W - 14} y={216} textAnchor="end" fontSize="10.5" fill="#6b7483">these add up to 1</text>
        )}

        {/* ---- the output ---- */}
        {step >= 7 && (
          <g opacity={step === 7 ? appear(e, 0.45, 0.4) : 1}>
            <rect x={qx - 70} y={388} width={140} height={30} rx={6}
              fill="rgba(251,191,36,0.1)" stroke={colC} />
            <text x={qx} y={407} textAnchor="middle" fontSize="11.5" fill={colC}>new vector for "mat"</text>
            <Strip v={data.out[Q_IDX]} x={qx - 38} y={372} color={colC} />
          </g>
        )}

        {/* ---- the whole matrix ---- */}
        {step === 8 && (
          <g opacity={appear(e, 0, 0.25)}>
            <rect x={0} y={0} width={W} height={H} fill="#0f1319" opacity={0.93} />
            <text x={W / 2} y={44} textAnchor="middle" fontSize="13" fill="#c7cdd8">
              every query against every key, in one matrix
            </text>
            {data.weights.map((row, i) =>
              row.map((v, j) => {
                const idx = i * 6 + j
                const on = stagger(e, idx, 36, 0.93)
                const blocked = j > i
                return (
                  <g key={`${i}-${j}`} opacity={on}>
                    <rect x={248 + j * 46} y={72 + i * 46} width={44} height={44} rx={3}
                      fill={blocked ? '#15181d' : colA} fillOpacity={blocked ? 1 : 0.1 + v * 0.9} />
                    <text x={270 + j * 46} y={99 + i * 46} textAnchor="middle" fontSize="9.5"
                      fontFamily="ui-monospace" fill={blocked ? '#3a424e' : v > 0.45 ? '#06201c' : '#c7cdd8'}>
                      {blocked ? '' : v.toFixed(2)}
                    </text>
                  </g>
                )
              }))}
            {TOKENS.map((tk, i) => (
              <g key={i} fontSize="10" fill="#6b7483" fontFamily="ui-monospace">
                <text x={240} y={99 + i * 46} textAnchor="end">{tk}</text>
                <text x={270 + i * 46} y={64} textAnchor="middle">{tk}</text>
              </g>
            ))}
            <text x={130} y={200} textAnchor="middle" fontSize="10.5" fill="#6b7483">asking ↓</text>
            <text x={130} y={218} textAnchor="middle" fontSize="10.5" fill="#6b7483">looking at →</text>
          </g>
        )}
      </svg>
    )
  }

  return <Scene steps={STEPS} height={430} autoPlay>{draw}</Scene>
}
