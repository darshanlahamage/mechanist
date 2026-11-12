import { useMemo, useState } from 'react'
import { HARDWARE, PRESETS, decodeStep, fmtBytes, fmtNum, kvCacheBytes, params, prefill } from '../lib/cost'
import { Lab, Line, Panel, Select, Slider, Stat } from '../components/ui'

export default function CostLab() {
  const [modelName, setModelName] = useState(Object.keys(PRESETS)[1])
  const [hwName, setHwName] = useState(HARDWARE[0].name)
  const [seqLen, setSeqLen] = useState(2048)
  const [batch, setBatch] = useState(1)

  const m = PRESETS[modelName]
  const hw = HARDWARE.find(h => h.name === hwName)!
  const p = params(m)

  const dec = decodeStep(m, seqLen, batch, hw)
  const pre = prefill(m, seqLen, batch, hw)
  const kv = kvCacheBytes(m, seqLen, batch)
  const weights = p.total * (m.dtype === 'fp32' ? 4 : m.dtype === 'int8' ? 1 : 2)
  const used = weights + kv
  const fits = used < hw.memory

  const throughput = useMemo(() => {
    const pts: [number, number][] = []
    for (let b = 1; b <= 256; b = b < 8 ? b + 1 : Math.round(b * 1.3)) {
      pts.push([b, decodeStep(m, seqLen, b, hw).tokensPerSec])
    }
    return pts
  }, [m, seqLen, hw])

  const maxBatch = Math.max(1, Math.floor((hw.memory * 0.9 - weights) / (kv / batch)))

  return (
    <Lab title="What does running this model actually cost?"
      takeaway={<>Generating one token means reading <b>every weight</b> out of memory to do only two
        floating point operations per weight. That is why a single request barely uses the GPU's math
        units, and why batching is the only real lever: the weights get read once and serve the whole
        batch. Watch the tokens-per-second curve flatten — that is the point where you stop being
        limited by memory speed and start being limited by math.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Setup">
            <div className="space-y-3">
              <Select label="model" value={modelName} options={Object.keys(PRESETS)} onChange={setModelName} />
              <Select label="hardware" value={hwName} options={HARDWARE.map(h => h.name)} onChange={setHwName} />
              <Slider label="sequence length" min={128} max={32768} step={128} value={seqLen}
                      onChange={setSeqLen} fmt={v => v.toLocaleString()} />
              <Slider label="batch size" min={1} max={256} value={batch} onChange={setBatch} />
            </div>
          </Panel>
          <Stat label="parameters" value={fmtNum(p.total)} sub={`${fmtBytes(weights)} in ${m.dtype}`} />
          <Stat label="fits in memory?" value={fits ? 'yes' : 'no'}
                sub={`${fmtBytes(used)} of ${fmtBytes(hw.memory)}`} tone={fits ? 'good' : 'bad'} />
          <Stat label="max batch at this length" value={String(maxBatch)} sub="before memory runs out" />
        </div>

        <div className="space-y-4">
          <Panel title="One decode step — generating a single token">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <Stat label="time" value={`${dec.timeMs.toFixed(1)} ms`} />
              <Stat label="tokens/sec" value={dec.tokensPerSec.toFixed(1)} tone="good" />
              <Stat label="limited by" value={dec.bound} tone={dec.bound === 'memory' ? 'warn' : 'good'} />
              <Stat label="math per byte" value={dec.intensity.toFixed(2)}
                    sub={`hardware wants ${dec.ridgePoint.toFixed(0)}`} tone={dec.intensity < dec.ridgePoint ? 'warn' : 'good'} />
            </div>
            <div className="mt-3 text-[12.5px] leading-relaxed text-[var(--dim)]">
              Memory traffic this step: <b className="text-[#3e392f]">{fmtBytes(dec.weightBytes)}</b> of weights
              + <b className="text-[#3e392f]">{fmtBytes(dec.kvBytes)}</b> of KV cache.
              The GPU can do {fmtNum(dec.ridgePoint)} operations in the time it takes to read one byte,
              and this step only does {dec.intensity.toFixed(2)}. So {' '}
              {dec.bound === 'memory'
                ? <>it sits idle waiting for memory — <b className="text-[var(--warn)]">{((1 - dec.intensity / dec.ridgePoint) * 100).toFixed(1)}% of the math units do nothing</b>.</>
                : <>the math units are now the limit.</>}
            </div>
          </Panel>

          <Panel title="Tokens per second as you add more requests">
            <Line series={[{ name: 'throughput', color: '#0b6f66', points: throughput }]}
                  markers={[{ x: batch, label: 'you' }]}
                  xLabel="batch size" yLabel="tokens/sec" height={160} />
          </Panel>

          <div className="grid gap-3 sm:grid-cols-2">
            <Panel title="Prefill — reading the prompt">
              <div className="space-y-2">
                <Stat label="time to first token" value={`${pre.timeMs.toFixed(0)} ms`} />
                <Stat label="limited by" value={pre.bound} tone={pre.bound === 'compute' ? 'good' : 'warn'} />
                <Stat label="spent on attention" value={`${(pre.attnShare * 100).toFixed(1)}%`}
                      sub="grows with the square of length" tone={pre.attnShare > 0.3 ? 'warn' : 'normal'} />
              </div>
            </Panel>
            <Panel title="Where the KV cache goes">
              <div className="space-y-2">
                <Stat label="per sequence" value={fmtBytes(kv / batch)} />
                <Stat label="all requests" value={fmtBytes(kv)}
                      tone={kv > hw.memory * 0.5 ? 'bad' : 'normal'} />
                <div className="text-[12px] leading-relaxed text-[var(--dim)]">
                  {m.nKVHeads < m.nHeads
                    ? <>This model shares K and V across heads ({m.nHeads} query heads, {m.nKVHeads} KV heads).
                        Without that it would need {fmtBytes(kv * (m.nHeads / m.nKVHeads))}.</>
                    : <>Every head keeps its own K and V. Sharing them would cut this a lot.</>}
                </div>
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </Lab>
  )
}
