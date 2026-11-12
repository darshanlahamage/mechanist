import { useMemo, useState } from 'react'
import { BYTES, HARDWARE, PRESETS, fmtBytes, params } from '../lib/cost'
import { Lab, Panel, Select, Slider, Stat, Toggle } from '../components/ui'

type Zero = 'none' | 'ZeRO-1' | 'ZeRO-2' | 'ZeRO-3 / FSDP'

export default function TrainMemoryLab() {
  const [modelName, setModelName] = useState('Llama 3 8B')
  const [gpus, setGpus] = useState(8)
  const [batch, setBatch] = useState(4)
  const [seqLen, setSeqLen] = useState(4096)
  const [zero, setZero] = useState<Zero>('none')
  const [checkpoint, setCheckpoint] = useState(false)
  const [hwName, setHwName] = useState('H100 SXM')

  const m = PRESETS[modelName]
  const hw = HARDWARE.find(h => h.name === hwName)!
  const P = params(m).total

  const parts = useMemo(() => {
    const shard = (n: number, on: boolean) => (on ? n / gpus : n)
    const w = P * BYTES[m.dtype]
    const g = P * BYTES[m.dtype]
    const master = P * 4
    const optim = P * 8                    // Adam: two states in fp32

    const shardOpt = zero !== 'none'
    const shardGrad = zero === 'ZeRO-2' || zero === 'ZeRO-3 / FSDP'
    const shardWeight = zero === 'ZeRO-3 / FSDP'

    // activations: roughly 16 tensors the size of the hidden state per layer
    const actFull = batch * seqLen * m.dModel * m.nLayers * 16 * BYTES[m.dtype]
    const act = checkpoint ? actFull / Math.sqrt(m.nLayers) + batch * seqLen * m.dModel * 4 * BYTES[m.dtype] : actFull

    return [
      { name: 'weights', bytes: shard(w, shardWeight), color: '#0b6f66' },
      { name: 'gradients', bytes: shard(g, shardGrad), color: '#6a46a3' },
      { name: 'fp32 master copy', bytes: shard(master, shardOpt), color: '#2563a8' },
      { name: 'Adam states', bytes: shard(optim, shardOpt), color: '#9a5c05' },
      { name: 'activations', bytes: act, color: '#b03a31' },
    ]
  }, [P, m, gpus, batch, seqLen, zero, checkpoint])

  const total = parts.reduce((s, p) => s + p.bytes, 0)
  const fits = total < hw.memory * 0.92
  const inference = P * BYTES[m.dtype]

  return (
    <Lab title="Why training needs so much more memory than inference"
      takeaway={<>The weights are the small part. Adam keeps two extra numbers per parameter in
        fp32, plus a master copy, which is <b>12 bytes per parameter</b> on top of the 2 you thought
        you needed. Activations then scale with batch and sequence length and can dwarf everything
        else. ZeRO splits the optimizer state across GPUs; checkpointing trades compute to shrink
        activations. Turn both on and watch an impossible run become possible.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Setup">
            <div className="space-y-3">
              <Select label="model" value={modelName} options={Object.keys(PRESETS)} onChange={setModelName} />
              <Select label="GPU" value={hwName} options={HARDWARE.map(h => h.name)} onChange={setHwName} />
              <Slider label="GPUs" min={1} max={64} value={gpus} onChange={setGpus} />
              <Slider label="batch per GPU" min={1} max={32} value={batch} onChange={setBatch} />
              <Slider label="sequence length" min={512} max={32768} step={512} value={seqLen}
                      onChange={setSeqLen} fmt={v => v >= 1024 ? `${(v / 1024).toFixed(0)}k` : String(v)} />
              <Select label="sharding" value={zero}
                      options={['none', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3 / FSDP'] as const} onChange={setZero} />
              <Toggle label="activation checkpointing" value={checkpoint} onChange={setCheckpoint}
                      hint="recompute instead of storing, ~30% slower" />
            </div>
          </Panel>
          <Stat label="per GPU" value={fmtBytes(total)}
                sub={`${fmtBytes(hw.memory)} available`} tone={fits ? 'good' : 'bad'} />
          <Stat label="just for inference" value={fmtBytes(inference)}
                sub={`training needs ${(total / inference).toFixed(1)}× more`} />
          {!fits && (
            <div className="rounded-lg border-l-2 border-[var(--bad)] bg-[rgba(176,58,49,0.07)] px-3 py-2 text-[12px] leading-relaxed text-[#3e392f]">
              Out of memory. Shrink the batch, turn on checkpointing, or shard further with ZeRO-3.
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Panel title="Where the memory goes, per GPU">
            <div className="mb-4 flex h-9 w-full overflow-hidden rounded-md">
              {parts.map(p => (
                <div key={p.name} className="group relative h-full transition-all"
                     style={{ width: `${(p.bytes / total) * 100}%`, background: p.color }}>
                  <div className="pointer-events-none absolute bottom-full left-1/2 mb-1 hidden -translate-x-1/2 whitespace-nowrap rounded bg-[var(--text)] px-1.5 py-0.5 font-mono text-[10px] text-[#fffdf7] group-hover:block">
                    {p.name}: {fmtBytes(p.bytes)}
                  </div>
                </div>
              ))}
            </div>
            <table className="w-full font-mono text-[12px]">
              <tbody>
                {parts.map(p => (
                  <tr key={p.name} className="border-t border-[var(--line)]">
                    <td className="py-1.5">
                      <span className="mr-2 inline-block h-2 w-2 rounded-sm align-middle" style={{ background: p.color }} />
                      <span className="text-[#3e392f]">{p.name}</span>
                    </td>
                    <td className="py-1.5 text-right text-[var(--accent)]">{fmtBytes(p.bytes)}</td>
                    <td className="py-1.5 text-right text-[var(--dim)]">{((p.bytes / total) * 100).toFixed(0)}%</td>
                  </tr>
                ))}
                <tr className="border-t border-[var(--line)]">
                  <td className="py-1.5 font-semibold text-[var(--text)]">total</td>
                  <td className={`py-1.5 text-right font-semibold ${fits ? 'text-[var(--accent)]' : 'text-[var(--bad)]'}`}>{fmtBytes(total)}</td>
                  <td />
                </tr>
              </tbody>
            </table>
          </Panel>

          <Panel title="The arithmetic, spelled out">
            <div className="font-mono text-[12px] leading-[1.9] text-[var(--dim)]">
              parameters = <span className="text-[var(--accent)]">{(P / 1e9).toFixed(2)}B</span><br />
              weights = P × {BYTES[m.dtype]} bytes ({m.dtype})<br />
              gradients = P × {BYTES[m.dtype]} bytes<br />
              fp32 master copy = P × 4 bytes<br />
              Adam momentum + variance = P × 8 bytes<br />
              <span className="text-[#3e392f]">→ {(2 + 2 + 4 + 8)} bytes per parameter before a single activation</span>
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
