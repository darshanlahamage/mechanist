import CostLab from '../../labs/CostLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const decodeCost: Concept = {
  id: 'decode-cost',
  title: 'Why generating tokens is slow',
  tagline: 'A decode step reads every weight in the model to do two operations with each one. The GPU is almost entirely idle.',
  track: 'inference',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  prereqs: ['attention'],
  papers: [
    { title: 'Efficient Memory Management for LLM Serving with PagedAttention', year: 2023,
      url: 'https://arxiv.org/abs/2309.06180',
      idea: 'Treat the KV cache like virtual memory pages. Stops fragmentation and lets many more requests share a GPU.' },
    { title: 'Fast Transformer Decoding: One Write-Head is All You Need', year: 2019,
      url: 'https://arxiv.org/abs/1911.02150',
      idea: 'Share one key/value head across all query heads. Cuts the cache dramatically with little quality loss.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>People assume a big model is slow because there is a lot of math. That is not the reason.</p>
        <p>To produce one token, the GPU has to read every weight in the model out of memory. For an
        8B model in bf16 that is 16 GB of reading. With each weight it does two operations: a multiply
        and an add. Two operations per two bytes.</p>
        <p>Modern GPUs can do several hundred operations in the time it takes to fetch one byte. So
        the math units sit there doing nothing while the memory system crawls. You are not compute
        limited. You are limited by memory bandwidth, and nothing else.</p>
        <Note kind="key" title="The useful consequence">
          <p>If the weights have to be read anyway, read them once and use them for many requests at
          the same time. That is batching, and it is why serving 64 users at once costs barely more
          than serving one.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Generation has two very different phases.</p>
        <h3>Prefill — reading the prompt</h3>
        <p>All prompt tokens go through at once. Every matmul has a real batch dimension, so the GPU
        is busy. This is compute bound and it decides your time to first token.</p>
        <h3>Decode — producing tokens one at a time</h3>
        <p>Only one new token exists, so every matrix multiply becomes a matrix-vector multiply. The
        shapes collapse and there is nothing to keep the hardware busy. This is memory bound and it
        decides how fast text streams out.</p>
        <p>Change the model, hardware and batch size, and watch where the limit moves:</p>
        <CostLab />
        <h3>The number that explains everything</h3>
        <p>Arithmetic intensity is operations done per byte read:</p>
        <Eq block tex="I = \frac{\text{FLOPs}}{\text{bytes moved}}" />
        <p>An H100 does about 989 TFLOP/s in bf16 with 3.35 TB/s of bandwidth. The ratio is about
        <b> 295 operations per byte</b>. A batch-1 decode step achieves roughly <b>1</b>. You are
        using about a third of one percent of the available math.</p>
      </>
    ),
    math: () => (
      <>
        <h3>Decode time from first principles</h3>
        <p>Bytes that must move for one step, with <Eq tex="N" /> parameters and batch
        <Eq tex="B" />:</p>
        <Eq block tex="\text{bytes} = N \cdot b_w + 2 \cdot L \cdot H_{kv} \cdot d_h \cdot S \cdot B \cdot b_{kv}" />
        <p>First term is the weights, read once regardless of batch size. Second is the KV cache, read
        once per sequence. Operations done:</p>
        <Eq block tex="\text{FLOPs} \approx 2NB" />
        <p>So time per step, when memory bound, is simply:</p>
        <Eq block tex="t = \frac{\text{bytes}}{\text{bandwidth}}, \qquad \text{tokens/sec} = \frac{B}{t}" />
        <p>Try it: 8B parameters in bf16 is 16 GB. On an H100 at 3.35 TB/s that is 4.8 ms per step,
        about <b>210 tokens/sec</b> at batch 1. Real systems land near this, which is a good sign the
        model is right.</p>

        <h3>Where batching stops helping</h3>
        <p>Weights are amortised across the batch, but the KV cache is not — every sequence has its
        own. Intensity grows with batch until either the KV traffic dominates, or you pass the
        hardware's ratio and become compute bound:</p>
        <Eq block tex="B^* \approx \frac{\text{peak FLOP/s}}{\text{bandwidth}} \cdot \frac{b_w}{2}" />
        <p>For an H100 in bf16 that is roughly 295 sequences, in theory. In practice memory for the
        KV cache runs out long before that, which is exactly the problem PagedAttention was built to
        solve.</p>

        <h3>Why GQA was invented</h3>
        <p>KV cache size scales with the number of KV heads. Llama 3 8B has 32 query heads but only 8
        KV heads, so its cache is 4× smaller than full multi-head attention would give. At 32k context
        that is the difference between fitting 4 requests on a GPU and fitting 16.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`def decode_tokens_per_second(n_params, bytes_per_param, bandwidth, kv_bytes=0, batch=1):
    """The whole model, in four lines."""
    bytes_moved = n_params * bytes_per_param + kv_bytes * batch
    seconds_per_step = bytes_moved / bandwidth
    return batch / seconds_per_step

# Llama 3 8B, bf16, on an H100
print(decode_tokens_per_second(8e9, 2, 3.35e12))     # ~209 tok/s`}</Code>
        <Code lang="python">{`def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch, dtype_bytes=2):
    # 2 = one for K, one for V
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes

# Llama 3 8B at 32k context, one request
print(kv_cache_bytes(32, 8, 128, 32768, 1) / 1e9)    # 4.3 GB -- for ONE user`}</Code>
        <Note kind="info" title="Measure it rather than trusting the number">
          <p>Run a real benchmark and compare. If you are getting far less than this model predicts,
          something is wrong: launch overhead between tiny kernels, a Python loop in the sampling path,
          or weights that are not actually in the dtype you think.</p>
        </Note>
        <Code lang="python">{`# what the GPU is really doing, per step
import torch
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50):
    out = model(input_ids, past_key_values=cache, use_cache=True)
torch.cuda.synchronize()
print((time.perf_counter() - t0) / 50 * 1000, "ms/token")`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>The levers, in order of how much they buy you</h3>
        <table className="w-full text-[13px]">
          <thead><tr className="text-[var(--dim)]">
            <th className="py-2 text-left font-normal">Lever</th>
            <th className="py-2 text-left font-normal">What it does</th>
            <th className="py-2 text-left font-normal">Catch</th>
          </tr></thead>
          <tbody>
            {[['Batching', 'Weights read once for many requests. Near-linear throughput gain.', 'Needs KV cache memory; adds latency per request.'],
              ['Quantisation', 'Fewer bytes per weight means directly fewer bytes read. int8 is ~2x.', 'Quality loss; dequantisation must be fused or you lose the gain.'],
              ['GQA / MLA', 'Shrinks the KV cache, so more requests fit.', 'Architectural — decided at training time.'],
              ['Paged KV cache', 'Removes fragmentation, often 2-4x more concurrent requests.', 'Needs a serving stack built for it.'],
              ['Speculative decoding', 'Verify several draft tokens in one pass, using idle math units.', 'Only helps when the draft is accepted often.'],
              ['KV cache quantisation', 'int8 cache halves the per-sequence cost at long context.', 'Accuracy drop grows with context length.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-3 font-medium text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 pr-3 text-[var(--dim)]">{r[1]}</td>
                <td className="py-2 text-[var(--dim)]">{r[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Why speculative decoding works at all</h3>
        <p>It looks like cheating: run a small model to guess several tokens, then check them all with
        the big model in one pass. The reason it is free is that the verification pass has a batch
        dimension again, and you had spare math units anyway. You are paying with compute you were
        not using.</p>
        <h3>Prefill and decode want different machines</h3>
        <p>Prefill is compute bound and wants raw FLOPs. Decode is memory bound and wants bandwidth
        and capacity. Running both on the same GPU means one phase constantly interrupts the other.
        That is the argument behind disaggregated serving — separate pools, each sized for its own
        bottleneck, with the KV cache shipped between them.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <thead><tr className="text-[var(--dim)]">
            <th className="py-2 text-left font-normal">Symptom</th>
            <th className="py-2 text-left font-normal">Cause</th>
          </tr></thead>
          <tbody>
            {[['Throughput far below the bandwidth estimate', 'Kernel launch overhead on tiny decode kernels. Use CUDA graphs.'],
              ['Adding GPUs barely helped', 'Tensor parallel adds a synchronisation every layer. Decode is latency bound, not compute bound.'],
              ['Out of memory at long context, fine when short', 'KV cache, not weights. It grows linearly with every token generated.'],
              ['Throughput collapses under load', 'KV fragmentation. Requests cannot be admitted even though memory is technically free.'],
              ['First token fast, streaming slow', 'Normal and expected. Prefill and decode are different workloads.'],
              ['int8 weights gave no speedup', 'Dequantisation running as a separate kernel, so you read more bytes, not fewer.'],
              ['Speculative decoding made it slower', 'Acceptance rate too low. Every rejection wastes the draft work.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Pick Llama 3 70B on an RTX 4090. It does not fit. Now look at the max batch number for
          8B at 32k context — it is small, and the KV cache is the reason, not the weights.</p>
        </Note>
      </>
    ),
  },
}
