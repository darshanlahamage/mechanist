import { useState } from 'react'
import { BYTES, type DType, fmtBytes } from '../lib/cost'
import { Lab, Panel, Select, Slider, Stat } from '../components/ui'

type Scheme = { name: string; kvHeads: (h: number) => number; latent?: number; note: string }

const SCHEMES: Scheme[] = [
  { name: 'MHA', kvHeads: h => h, note: 'Every query head keeps its own K and V. Original transformer.' },
  { name: 'GQA (8 groups)', kvHeads: h => Math.min(8, h), note: 'Heads share K and V in groups. Llama 3, Qwen, Mistral.' },
  { name: 'GQA (2 groups)', kvHeads: h => Math.min(2, h), note: 'Aggressive sharing. Smaller cache, more quality risk.' },
  { name: 'MQA', kvHeads: () => 1, note: 'One K and V for all heads. Smallest cache, biggest quality hit.' },
  { name: 'MLA (latent)', kvHeads: h => h, latent: 512, note: 'Store a compressed latent instead of K and V. DeepSeek V2/V3.' },
]

export default function KVBudgetLab() {
  const [nHeads, setNHeads] = useState(32)
  const [dHead, setDHead] = useState(128)
  const [nLayers, setNLayers] = useState(32)
  const [seqLen, setSeqLen] = useState(8192)
  const [batch, setBatch] = useState(16)
  const [dtype, setDtype] = useState<DType>('bf16')

  const bytesFor = (s: Scheme) => s.latent
    ? nLayers * s.latent * seqLen * batch * BYTES[dtype]
    : 2 * nLayers * s.kvHeads(nHeads) * dHead * seqLen * batch * BYTES[dtype]

  const all = SCHEMES.map(s => ({ ...s, bytes: bytesFor(s) }))
  const max = Math.max(...all.map(a => a.bytes))
  const mha = all[0].bytes
  const gpuMem = 80e9

  return (
    <Lab title="What each attention scheme costs you in cache"
      takeaway={<>The KV cache is what decides how many users fit on one GPU, not the weights.
        Going from MHA to 8-group GQA cuts it by four with barely any quality loss, which is why
        almost every model released since 2023 uses it. MLA goes further by storing a compressed
        vector instead of the keys and values themselves.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Model and load">
            <div className="space-y-3">
              <Slider label="query heads" min={8} max={128} step={8} value={nHeads} onChange={setNHeads} />
              <Slider label="head dim" min={64} max={256} step={8} value={dHead} onChange={setDHead} />
              <Slider label="layers" min={8} max={126} step={2} value={nLayers} onChange={setNLayers} />
              <Slider label="context length" min={512} max={131072} step={512} value={seqLen}
                      onChange={setSeqLen} fmt={v => v >= 1024 ? `${(v / 1024).toFixed(0)}k` : String(v)} />
              <Slider label="concurrent requests" min={1} max={128} value={batch} onChange={setBatch} />
              <Select label="cache dtype" value={dtype} options={['bf16', 'fp16', 'fp8', 'int4'] as const}
                      onChange={setDtype} />
            </div>
          </Panel>
          <Stat label="MHA would need" value={fmtBytes(mha)}
                tone={mha > gpuMem ? 'bad' : 'normal'} sub="an 80 GB GPU holds 80 GB" />
        </div>

        <div className="space-y-4">
          <Panel title="KV cache size, same workload">
            <div className="space-y-3">
              {all.map(s => (
                <div key={s.name}>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[13px] font-medium text-[#3e392f]">{s.name}</span>
                    <span className="font-mono text-[12px] text-[var(--accent)]">
                      {fmtBytes(s.bytes)}
                      {s.bytes < mha && <span className="ml-2 text-[var(--dim)]">{(mha / s.bytes).toFixed(1)}× smaller</span>}
                    </span>
                  </div>
                  <div className="mt-1 h-[10px] overflow-hidden rounded-sm bg-[var(--panel-2)]">
                    <div className="h-full rounded-sm" style={{ width: `${(s.bytes / max) * 100}%`,
                      background: s.bytes > gpuMem ? 'var(--bad)' : 'var(--accent)' }} />
                  </div>
                  <div className="mt-1 text-[11.5px] text-[var(--dim)]">{s.note}</div>
                </div>
              ))}
            </div>
            <div className="mt-4 border-t border-[var(--line)] pt-3 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Red bars do not fit in an 80 GB GPU at all, before you have loaded a single weight.
              Try 128k context with 32 requests on MHA and see how fast this becomes impossible.
            </div>
          </Panel>

          <Panel title="How many requests fit in 80 GB of cache">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {all.map(s => {
                const perReq = s.bytes / batch
                const fit = Math.floor(gpuMem / perReq)
                return <Stat key={s.name} label={s.name} value={String(fit)}
                             sub={`${fmtBytes(perReq)} each`} tone={fit < 4 ? 'bad' : fit > 32 ? 'good' : 'normal'} />
              })}
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
