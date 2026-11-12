// One cost model, used by every page. A number cannot be right in one place
// and wrong in another if it only exists here.

export type DType = 'fp32' | 'bf16' | 'fp16' | 'fp8' | 'int8' | 'int4'

export const BYTES: Record<DType, number> = {
  fp32: 4, bf16: 2, fp16: 2, fp8: 1, int8: 1, int4: 0.5,
}

export interface Hardware {
  name: string
  flops: Partial<Record<DType, number>>  // dense FLOP/s
  bandwidth: number                       // bytes/s off-chip memory
  memory: number                          // bytes of memory
  note: string
}

// Vendor spec-sheet peak numbers. Peak is never reached in practice —
// the labs show the gap instead of hiding it.
export const HARDWARE: Hardware[] = [
  { name: 'H100 SXM', flops: { bf16: 989e12, fp16: 989e12, fp8: 1979e12, fp32: 67e12 },
    bandwidth: 3350e9, memory: 80e9, note: 'Hopper, HBM3' },
  { name: 'A100 80GB', flops: { bf16: 312e12, fp16: 312e12, fp32: 19.5e12 },
    bandwidth: 2039e9, memory: 80e9, note: 'Ampere, HBM2e' },
  { name: 'L40S', flops: { bf16: 362e12, fp16: 362e12, fp8: 733e12, fp32: 91.6e12 },
    bandwidth: 864e9, memory: 48e9, note: 'Ada, GDDR6' },
  { name: 'RTX 4090', flops: { bf16: 165e12, fp16: 165e12, fp8: 330e12, fp32: 82.6e12 },
    bandwidth: 1008e9, memory: 24e9, note: 'Ada, consumer' },
  { name: 'M3 Max (40c)', flops: { fp16: 28e12, fp32: 14e12 },
    bandwidth: 400e9, memory: 128e9, note: 'Apple, unified memory' },
  { name: 'TPU v5e', flops: { bf16: 197e12, int8: 394e12 },
    bandwidth: 819e9, memory: 16e9, note: 'Google, systolic array' },
]

export interface ModelSpec {
  nLayers: number
  dModel: number
  nHeads: number
  nKVHeads: number
  dFFN: number
  vocab: number
  dtype: DType
  kvDtype: DType
}

export const PRESETS: Record<string, ModelSpec> = {
  'GPT-2 small (124M)': { nLayers: 12, dModel: 768, nHeads: 12, nKVHeads: 12, dFFN: 3072, vocab: 50257, dtype: 'fp16', kvDtype: 'fp16' },
  'Llama 3 8B':        { nLayers: 32, dModel: 4096, nHeads: 32, nKVHeads: 8, dFFN: 14336, vocab: 128256, dtype: 'bf16', kvDtype: 'bf16' },
  'Llama 3 70B':       { nLayers: 80, dModel: 8192, nHeads: 64, nKVHeads: 8, dFFN: 28672, vocab: 128256, dtype: 'bf16', kvDtype: 'bf16' },
  'Qwen3 4B':          { nLayers: 36, dModel: 2560, nHeads: 32, nKVHeads: 8, dFFN: 9728, vocab: 151936, dtype: 'bf16', kvDtype: 'bf16' },
  'Mistral 7B':        { nLayers: 32, dModel: 4096, nHeads: 32, nKVHeads: 8, dFFN: 14336, vocab: 32000, dtype: 'bf16', kvDtype: 'bf16' },
}

const dHead = (m: ModelSpec) => m.dModel / m.nHeads

/** Parameter count, split by where the weights actually sit. */
export function params(m: ModelSpec) {
  const dh = dHead(m)
  const attn = m.dModel * m.dModel          // Q
    + m.dModel * (m.nKVHeads * dh) * 2      // K and V (smaller when GQA)
    + m.dModel * m.dModel                   // output projection
  const ffn = 3 * m.dModel * m.dFFN         // SwiGLU: gate, up, down
  const perLayer = attn + ffn
  const embed = m.vocab * m.dModel
  return { attn, ffn, perLayer, blocks: perLayer * m.nLayers, embed, total: perLayer * m.nLayers + embed }
}

/** KV cache bytes for one sequence. The thing that actually limits batch size. */
export function kvCacheBytes(m: ModelSpec, seqLen: number, batch = 1) {
  const dh = dHead(m)
  return 2 * m.nLayers * m.nKVHeads * dh * seqLen * batch * BYTES[m.kvDtype]
}

/** Work for ONE decode step: read all the weights, do ~2 FLOPs per weight. */
export function decodeStep(m: ModelSpec, seqLen: number, batch: number, hw: Hardware) {
  const p = params(m)
  const weightBytes = p.blocks * BYTES[m.dtype]
  const kvBytes = kvCacheBytes(m, seqLen, batch)
  const bytes = weightBytes + kvBytes
  const flops = 2 * p.blocks * batch + 4 * m.nLayers * m.dModel * seqLen * batch
  const peak = hw.flops[m.dtype] ?? hw.flops.bf16 ?? hw.flops.fp16 ?? 1
  const tCompute = flops / peak
  const tMemory = bytes / hw.bandwidth
  const t = Math.max(tCompute, tMemory)
  return {
    flops, bytes, weightBytes, kvBytes,
    intensity: flops / bytes,
    ridgePoint: peak / hw.bandwidth,
    bound: tMemory > tCompute ? 'memory' : 'compute',
    timeMs: t * 1000,
    tokensPerSec: batch / t,
  } as const
}

/** Prefill: one big matmul-heavy pass over the whole prompt. */
export function prefill(m: ModelSpec, seqLen: number, batch: number, hw: Hardware) {
  const p = params(m)
  const dense = 2 * p.blocks * seqLen * batch
  const attnQuad = 4 * m.nLayers * m.dModel * seqLen * seqLen * batch
  const flops = dense + attnQuad
  const bytes = p.blocks * BYTES[m.dtype] + kvCacheBytes(m, seqLen, batch)
  const peak = hw.flops[m.dtype] ?? hw.flops.bf16 ?? hw.flops.fp16 ?? 1
  const tCompute = flops / peak, tMemory = bytes / hw.bandwidth
  const t = Math.max(tCompute, tMemory)
  return {
    flops, attnShare: attnQuad / flops, bytes,
    intensity: flops / bytes,
    bound: tMemory > tCompute ? 'memory' : 'compute',
    timeMs: t * 1000,
  } as const
}

/** Memory needed to TRAIN, broken into the four pieces people forget. */
export function trainingMemory(m: ModelSpec, batch: number, seqLen: number, opt: 'adamw' | 'sgd' = 'adamw') {
  const p = params(m).total
  const weights = p * BYTES[m.dtype]
  const grads = p * BYTES[m.dtype]
  const master = p * 4
  const optStates = opt === 'adamw' ? p * 8 : p * 4
  const acts = batch * seqLen * m.dModel * m.nLayers * 16 * BYTES[m.dtype]
  return { weights, grads, master, optStates, activations: acts,
    total: weights + grads + master + optStates + acts }
}

export function fmtBytes(b: number): string {
  if (!Number.isFinite(b)) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++ }
  return `${b.toFixed(b < 10 ? 2 : b < 100 ? 1 : 0)} ${u[i]}`
}

export function fmtNum(n: number): string {
  if (!Number.isFinite(n)) return '—'
  const a = Math.abs(n)
  if (a >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (a >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (a >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (a >= 1e3) return `${(n / 1e3).toFixed(2)}K`
  return n.toFixed(a < 10 ? 2 : 0)
}
