import { Scene, appear, ease, type Frame, type Step } from './Scene'

const W = 780, H = 400
const D = 64, BASE = 10000
const PAIRS = [0, 4, 12, 31]              // fast to slow, four of the 32 pairs

const STEPS: Step[] = [
  { label: 'A token vector, cut into pairs', say: 'Take the 64 numbers of one head and read them two at a time. Each pair is a point on a little plane.', hold: 3 },
  { label: 'Position turns into an angle', say: 'Move the token to position 1, 2, 3 and each pair rotates. Nothing is added to the vector — it is turned.', hold: 4.5 },
  { label: 'Every pair spins at its own speed', say: 'Pair 0 whips round in a few positions. Pair 31 has barely moved after thousands. Fast pairs encode "right next to me", slow ones encode "roughly where in the document".', hold: 4.5 },
  { label: 'Two tokens, two rotations', say: 'The query sits at position m, the key at position n. Both have been turned by their own angle.', hold: 3.2 },
  { label: 'The dot product only sees the gap', say: 'When you compare them, the shared rotation cancels. What is left depends on m − n alone. Absolute position went in, relative position came out.', hold: 4.5 },
  { label: 'Past the trained length', say: 'Push the position beyond anything seen in training and the fast pairs have spun through thousands of turns into angles the model has no behaviour for. That is what context extension methods slow back down.', hold: 4.5 },
]

const theta = (i: number) => Math.pow(BASE, (-2 * i) / D)

function Dial({ cx, cy, r, angle, label, sub, color = '#0b6f66', ghost }:
  { cx: number; cy: number; r: number; angle: number; label: string; sub?: string; color?: string; ghost?: number }) {
  const x = cx + r * Math.cos(angle), y = cy - r * Math.sin(angle)
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#dcd3c0" />
      {ghost !== undefined && (
        <line x1={cx} y1={cy} x2={cx + r * Math.cos(ghost)} y2={cy - r * Math.sin(ghost)}
          stroke="#cec4ae" strokeWidth="1.5" strokeDasharray="3 3" />
      )}
      <line x1={cx} y1={cy} x2={x} y2={y} stroke={color} strokeWidth="2.2" />
      <circle cx={x} cy={y} r="3.5" fill={color} />
      <text x={cx} y={cy + r + 17} textAnchor="middle" fontSize="10.5" fill="#7b7364">{label}</text>
      {sub && <text x={cx} y={cy + r + 30} textAnchor="middle" fontSize="9.5"
        fontFamily="ui-monospace" fill="#8f8775">{sub}</text>}
    </g>
  )
}

export default function RopeFlow() {
  const draw = ({ step, t }: Frame) => {
    const e = ease(t)

    // which position we are showing, per step
    const pos =
      step === 1 ? e * 6
      : step === 2 ? e * 240
      : step >= 3 ? 40
      : 0
    const farPos = step === 5 ? 512 + e * 15000 : 0

    const dials = PAIRS.map(i => ({ i, ang: pos * theta(i), wl: (2 * Math.PI) / theta(i) }))

    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: 'block' }}>
        {/* the vector, cut into pairs */}
        {step === 0 && (
          <g opacity={appear(e, 0, 0.25)}>
            <text x={W / 2} y={46} textAnchor="middle" fontSize="12" fill="#3e392f">
              one head of a token: 64 numbers
            </text>
            {Array.from({ length: 32 }, (_, i) => (
              <g key={i} opacity={appear(e, i / 60, 0.2)}>
                <rect x={60 + i * 20} y={66} width={8.5} height={22} rx={1.5} fill="#0b6f66" opacity={0.75} />
                <rect x={69.5 + i * 20} y={66} width={8.5} height={22} rx={1.5} fill="#0b6f66" opacity={0.4} />
                <rect x={59 + i * 20} y={64} width={20} height={26} rx={2.5} fill="none"
                  stroke="#dcd3c0" opacity={appear(e, 0.45, 0.3)} />
              </g>
            ))}
            <text x={W / 2} y={116} textAnchor="middle" fontSize="11" fill="#8a8171"
              opacity={appear(e, 0.5, 0.3)}>32 pairs — each one becomes a point on a plane</text>
          </g>
        )}

        {/* the dials */}
        {step >= 1 && step <= 2 && (
          <g>
            <text x={W / 2} y={44} textAnchor="middle" fontSize="12.5" fill="#3e392f">
              position {Math.round(pos)}
            </text>
            {dials.map((d, k) => (
              <Dial key={d.i} cx={140 + k * 168} cy={170} r={56} angle={d.ang}
                color={k === 0 ? '#0b6f66' : k === 3 ? '#6a46a3' : '#2f8f82'}
                label={`pair ${d.i}`}
                sub={step === 2
                  ? `${(d.ang / (2 * Math.PI)).toFixed(d.i > 20 ? 4 : 1)} turns`
                  : `every ${Math.round(d.wl)} positions`} />
            ))}
            {step === 2 && (
              <text x={W / 2} y={310} textAnchor="middle" fontSize="11.5" fill="#8a8171">
                same position, wildly different angles — that spread is what carries distance
              </text>
            )}
          </g>
        )}

        {/* query and key, and the cancellation */}
        {step >= 3 && step <= 4 && (
          <g>
            <text x={210} y={44} textAnchor="middle" fontSize="12" fill="#0b6f66">query at position 40</text>
            <text x={570} y={44} textAnchor="middle" fontSize="12" fill="#6a46a3">key at position 25</text>
            {PAIRS.slice(0, 3).map((i, k) => {
              const aq = 40 * theta(i), ak = 25 * theta(i)
              const gap = aq - ak
              return (
                <g key={i}>
                  <Dial cx={120 + k * 92} cy={140} r={38} angle={aq} label={`pair ${i}`} />
                  <Dial cx={480 + k * 92} cy={140} r={38} angle={ak} label={`pair ${i}`} color="#6a46a3" />
                  {step === 4 && (
                    <g opacity={appear(e, 0.15 + k * 0.12, 0.3)}>
                      <Dial cx={240 + k * 150} cy={296} r={40} angle={gap} label={`difference`}
                        sub={`${(gap / (2 * Math.PI)).toFixed(2)} turns`} color="#9a5c05" ghost={0} />
                    </g>
                  )}
                </g>
              )
            })}
            {step === 4 && (
              <text x={W / 2} y={386} textAnchor="middle" fontSize="12" fill="#9a5c05"
                opacity={appear(e, 0.5, 0.3)}>
                40 − 25 = 15, whatever the absolute positions were
              </text>
            )}
          </g>
        )}

        {/* running off the end */}
        {step === 5 && (
          <g opacity={appear(e, 0, 0.2)}>
            <text x={W / 2} y={46} textAnchor="middle" fontSize="12.5" fill="#3e392f">
              position {Math.round(farPos).toLocaleString()}
            </text>
            {PAIRS.map((i, k) => {
              const turns = (farPos * theta(i)) / (2 * Math.PI)
              const wild = turns > 60
              return (
                <Dial key={i} cx={140 + k * 168} cy={180} r={56} angle={farPos * theta(i)}
                  color={wild ? '#b03a31' : '#0b6f66'} label={`pair ${i}`}
                  sub={`${turns < 1 ? turns.toFixed(3) : Math.round(turns).toLocaleString()} turns`} />
              )
            })}
            <rect x={60} y={300} width={W - 120} height={2} fill="#dcd3c0" />
            <rect x={60} y={300} width={(W - 120) * 0.034} height={2} fill="#0b6f66" />
            <text x={60} y={322} fontSize="10.5" fill="#0b6f66">trained to here (512)</text>
            <text x={W - 60} y={322} textAnchor="end" fontSize="10.5" fill="#b03a31">
              {Math.round(farPos).toLocaleString()} — never seen
            </text>
          </g>
        )}
      </svg>
    )
  }

  return <Scene steps={STEPS} height={400} autoPlay>{draw}</Scene>
}
