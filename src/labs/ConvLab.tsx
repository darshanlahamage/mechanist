import { useEffect, useMemo, useRef, useState } from 'react'
import { Lab, Panel, Select, Slider, Stat } from '../components/ui'

const SIZE = 28

const KERNELS = {
  'edge detect': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
  'sobel x': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  'sobel y': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  'blur': [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  'sharpen': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  'identity': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
} as const
type KName = keyof typeof KERNELS

/** A small picture with a clear shape, so the filters obviously do something. */
function makeImage(): number[][] {
  const img: number[][] = []
  for (let y = 0; y < SIZE; y++) {
    const row: number[] = []
    for (let x = 0; x < SIZE; x++) {
      let v = 0.12
      if (x > 5 && x < 22 && y > 5 && y < 22) v = 0.8                      // square
      if ((x - 14) ** 2 + (y - 14) ** 2 < 20) v = 0.25                     // hole in the middle
      if (y > 23) v = 0.5 + 0.3 * Math.sin(x * 0.9)                        // texture strip
      row.push(v)
    }
    img.push(row)
  }
  return img
}

function Grid({ data, cell, highlight, onHover, label }: {
  data: number[][]; cell: number; label: string
  highlight?: { x: number; y: number; k: number } | null
  onHover?: (p: { x: number; y: number } | null) => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const rows = data.length, cols = data[0].length
  useEffect(() => {
    const cv = ref.current!
    const dpr = window.devicePixelRatio || 1
    cv.width = cols * cell * dpr; cv.height = rows * cell * dpr
    cv.style.width = `${cols * cell}px`; cv.style.height = `${rows * cell}px`
    const g = cv.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    let lo = Infinity, hi = -Infinity
    for (const r of data) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v }
    for (let y = 0; y < rows; y++)
      for (let x = 0; x < cols; x++) {
        const t = hi === lo ? 0.5 : (data[y][x] - lo) / (hi - lo)
        const c = Math.round(t * 255)
        g.fillStyle = `rgb(${c},${c},${c})`
        g.fillRect(x * cell, y * cell, cell, cell)
      }
    if (highlight) {
      g.strokeStyle = '#0b6f66'; g.lineWidth = 1.5
      g.strokeRect(highlight.x * cell, highlight.y * cell, highlight.k * cell, highlight.k * cell)
    }
  }, [data, cell, rows, cols, highlight])

  return (
    <div>
      <canvas ref={ref} className="rounded border border-[var(--line)]"
        style={{ imageRendering: 'pixelated', cursor: onHover ? 'crosshair' : 'default' }}
        onMouseMove={e => {
          if (!onHover) return
          const r = ref.current!.getBoundingClientRect()
          onHover({ x: Math.floor((e.clientX - r.left) / cell), y: Math.floor((e.clientY - r.top) / cell) })
        }}
        onMouseLeave={() => onHover?.(null)} />
      <div className="mt-1.5 text-[11px] text-[var(--dim)]">{label} <span className="font-mono">{rows}×{cols}</span></div>
    </div>
  )
}

export default function ConvLab() {
  const [kname, setKname] = useState<KName>('edge detect')
  const [stride, setStride] = useState(1)
  const [pad, setPad] = useState(1)
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)

  const img = useMemo(makeImage, [])
  const kernel = KERNELS[kname]
  const K = 3

  const out = useMemo(() => {
    const outSize = Math.floor((SIZE + 2 * pad - K) / stride) + 1
    const o: number[][] = []
    for (let oy = 0; oy < outSize; oy++) {
      const row: number[] = []
      for (let ox = 0; ox < outSize; ox++) {
        let s = 0
        for (let ky = 0; ky < K; ky++)
          for (let kx = 0; kx < K; kx++) {
            const iy = oy * stride + ky - pad, ix = ox * stride + kx - pad
            const v = iy >= 0 && iy < SIZE && ix >= 0 && ix < SIZE ? img[iy][ix] : 0
            s += v * kernel[ky][kx]
          }
        row.push(s)
      }
      o.push(row)
    }
    return o
  }, [img, kernel, stride, pad])

  const outSize = out.length
  const window = useMemo(() => {
    if (!hover) return null
    const oy = Math.min(hover.y, outSize - 1), ox = Math.min(hover.x, outSize - 1)
    const vals: number[][] = [], prods: number[][] = []
    let sum = 0
    for (let ky = 0; ky < K; ky++) {
      const vr: number[] = [], pr: number[] = []
      for (let kx = 0; kx < K; kx++) {
        const iy = oy * stride + ky - pad, ix = ox * stride + kx - pad
        const v = iy >= 0 && iy < SIZE && ix >= 0 && ix < SIZE ? img[iy][ix] : 0
        vr.push(v); pr.push(v * kernel[ky][kx]); sum += v * kernel[ky][kx]
      }
      vals.push(vr); prods.push(pr)
    }
    return { oy, ox, vals, prods, sum, ix: ox * stride - pad, iy: oy * stride - pad }
  }, [hover, img, kernel, stride, pad, outSize])

  const macs = outSize * outSize * K * K

  return (
    <Lab title="Convolution: one sliding window at a time"
      takeaway={<>A convolution is a tiny dot product repeated at every position. The same nine
        numbers are reused everywhere, which is the whole idea — a feature detector that works in
        the top left also works in the bottom right, and you only pay for nine weights. Hover the
        output to see the exact multiply-add that produced that pixel.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Select label="kernel" value={kname} options={Object.keys(KERNELS) as KName[]} onChange={setKname} />
              <Slider label="stride" min={1} max={4} value={stride} onChange={setStride} />
              <Slider label="padding" min={0} max={3} value={pad} onChange={setPad} />
            </div>
          </Panel>
          <Stat label="output size" value={`${outSize}×${outSize}`}
                sub={`(${SIZE} + 2·${pad} − ${K}) / ${stride} + 1`} />
          <Stat label="multiply-adds" value={macs.toLocaleString()} sub="for this one filter" />
          <Panel title="The kernel">
            <table className="font-mono text-[12px]">
              <tbody>
                {kernel.map((r, i) => (
                  <tr key={i}>{r.map((v, j) => (
                    <td key={j} className="w-[46px] border border-[var(--line)] py-1 text-center text-[var(--accent)]">
                      {Number.isInteger(v) ? v : v.toFixed(2)}
                    </td>))}</tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title="Input and output">
            <div className="flex flex-wrap items-start gap-6">
              <Grid data={img} cell={9} label="input"
                    highlight={window ? { x: window.ix, y: window.iy, k: K } : null} />
              <Grid data={out} cell={Math.max(4, Math.floor(252 / outSize))} label="output"
                    onHover={setHover} />
            </div>
            <div className="mt-3 text-[12.5px] text-[var(--dim)]">
              {window
                ? <>Output pixel <span className="font-mono text-[var(--accent)]">[{window.oy},{window.ox}]</span> comes from the highlighted square on the left.</>
                : <>Hover the output image to see which input pixels made each one.</>}
            </div>
          </Panel>

          {window && (
            <Panel title="The arithmetic behind that one pixel">
              <div className="flex flex-wrap items-center gap-5">
                <table className="font-mono text-[11px]">
                  <tbody>{window.vals.map((r, i) => (
                    <tr key={i}>{r.map((v, j) => (
                      <td key={j} className="w-[48px] border border-[var(--line)] py-1 text-center text-[#3e392f]">{v.toFixed(2)}</td>
                    ))}</tr>))}</tbody>
                </table>
                <span className="text-[var(--dim)]">×</span>
                <table className="font-mono text-[11px]">
                  <tbody>{kernel.map((r, i) => (
                    <tr key={i}>{r.map((v, j) => (
                      <td key={j} className="w-[48px] border border-[var(--line)] py-1 text-center text-[var(--accent-2)]">
                        {Number.isInteger(v) ? v : v.toFixed(2)}</td>
                    ))}</tr>))}</tbody>
                </table>
                <span className="text-[var(--dim)]">=</span>
                <table className="font-mono text-[11px]">
                  <tbody>{window.prods.map((r, i) => (
                    <tr key={i}>{r.map((v, j) => (
                      <td key={j} className="w-[48px] border border-[var(--line)] py-1 text-center text-[var(--accent)]">{v.toFixed(2)}</td>
                    ))}</tr>))}</tbody>
                </table>
                <div className="font-mono text-[13px]">
                  <span className="text-[var(--dim)]">sum = </span>
                  <span className="text-[var(--accent)]">{window.sum.toFixed(3)}</span>
                </div>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </Lab>
  )
}
