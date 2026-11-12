import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'

export interface Step {
  label: string          // short title shown on the timeline
  say: string            // the sentence that explains what is happening right now
  hold?: number          // seconds this step runs (default 2.2)
}

export interface Frame {
  step: number           // which step we are in
  t: number              // 0 to 1 within this step
  global: number         // 0 to 1 across the whole animation
}

/**
 * The player every animation on the site uses.
 * It owns time; the animation only draws a frame. Play, pause, step forward and
 * back, scrub, change speed, loop. Nothing is pre-rendered — each frame is drawn
 * from the real numbers for that moment.
 */
export function Scene({ steps, children, height, autoPlay = false, loop = true, caption = true }: {
  steps: Step[]
  children: (f: Frame) => ReactNode
  height?: number
  autoPlay?: boolean
  loop?: boolean
  caption?: boolean
}) {
  const [playing, setPlaying] = useState(autoPlay)
  const [speed, setSpeed] = useState(1)
  const [elapsed, setElapsed] = useState(0)
  const raf = useRef<number>(0)
  const last = useRef<number>(0)

  const holds = steps.map(s => s.hold ?? 2.2)
  const total = holds.reduce((a, b) => a + b, 0)

  useEffect(() => {
    if (!playing) return
    last.current = performance.now()
    const tick = (now: number) => {
      const dt = (now - last.current) / 1000
      last.current = now
      setElapsed(e => {
        const next = e + dt * speed
        if (next >= total) {
          if (loop) return 0
          setPlaying(false)
          return total
        }
        return next
      })
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, speed, total, loop])

  // turn elapsed seconds into a step index and a position inside that step
  let acc = 0, step = 0, t = 0
  for (let i = 0; i < holds.length; i++) {
    if (elapsed < acc + holds[i] || i === holds.length - 1) {
      step = i
      t = Math.min(1, (elapsed - acc) / holds[i])
      break
    }
    acc += holds[i]
  }

  const startOf = useCallback((i: number) => holds.slice(0, i).reduce((a, b) => a + b, 0), [steps])
  const goto = (i: number) => {
    const clamped = Math.max(0, Math.min(steps.length - 1, i))
    setElapsed(startOf(clamped) + 0.001)
  }

  return (
    <figure className="my-7 overflow-hidden rounded-xl border border-[#2b3340] bg-[#0f1319]">
      <div className="relative" style={{ minHeight: height }}>
        {children({ step, t, global: total ? elapsed / total : 0 })}
      </div>

      {caption && (
        <div className="border-t border-[#20262f] px-4 py-3">
          <div className="flex items-baseline gap-2">
            <span className="shrink-0 font-mono text-[11px] text-[var(--accent)]">
              {step + 1}/{steps.length}
            </span>
            <span className="text-[13.5px] font-medium text-white">{steps[step]?.label}</span>
          </div>
          <p className="mt-1 text-[13px] leading-relaxed text-[#b4bcc8]">{steps[step]?.say}</p>
        </div>
      )}

      <div className="flex items-center gap-2 border-t border-[#20262f] bg-[#0b0e13] px-3 py-2">
        <button onClick={() => goto(step - 1)} title="previous step"
          className="grid h-7 w-7 place-items-center rounded text-[var(--dim)] hover:bg-white/5 hover:text-white">◀</button>
        <button onClick={() => { if (elapsed >= total) setElapsed(0); setPlaying(p => !p) }}
          className="grid h-7 w-7 place-items-center rounded bg-[var(--accent)] text-[#06201c]">
          {playing ? '❚❚' : '▶'}
        </button>
        <button onClick={() => goto(step + 1)} title="next step"
          className="grid h-7 w-7 place-items-center rounded text-[var(--dim)] hover:bg-white/5 hover:text-white">▶</button>

        <div className="mx-1 flex flex-1 gap-[3px]">
          {steps.map((s, i) => (
            <button key={i} onClick={() => goto(i)} title={s.label}
              className="group relative h-6 flex-1 rounded-sm">
              <span className="absolute inset-x-0 top-1/2 h-[5px] -translate-y-1/2 overflow-hidden rounded-full bg-[#232a34]">
                <span className="block h-full rounded-full bg-[var(--accent)] transition-[width] duration-75"
                  style={{ width: i < step ? '100%' : i === step ? `${t * 100}%` : '0%' }} />
              </span>
            </button>
          ))}
        </div>

        <button onClick={() => setSpeed(s => (s === 1 ? 0.35 : s === 0.35 ? 2 : 1))}
          className="rounded px-1.5 py-0.5 font-mono text-[11px] text-[var(--dim)] hover:bg-white/5 hover:text-white">
          {speed}×
        </button>
        <button onClick={() => { setElapsed(0); setPlaying(false) }}
          className="rounded px-1.5 py-0.5 text-[11px] text-[var(--dim)] hover:bg-white/5 hover:text-white">reset</button>
      </div>
    </figure>
  )
}

/* ---- small helpers animations use constantly ---- */

/** Smooth 0→1. Everything moves with this so nothing snaps. */
export const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2)

/** Turn progress into "how far along is item i of n", staggered. */
export function stagger(t: number, i: number, n: number, overlap = 0.55) {
  const span = 1 / (n - (n - 1) * overlap || 1)
  const start = i * span * (1 - overlap)
  return Math.max(0, Math.min(1, (t - start) / span))
}

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t

/** Fade a thing in during the first part of a step and hold it after. */
export const appear = (t: number, at = 0, over = 0.3) =>
  Math.max(0, Math.min(1, (t - at) / over))
