import { useMemo, useState } from 'react'
import { SAMPLE_CORPUS, encodeBPE, trainBPE } from '../lib/bpe'
import { Lab, Panel, Slider, Stat } from '../components/ui'

function Tok({ t }: { t: string }) {
  const end = t.endsWith('</w>')
  const text = end ? t.slice(0, -4) : t
  return (
    <span className={`mr-[3px] inline-block rounded px-1.5 py-0.5 font-mono text-[11.5px] ${
      end ? 'bg-[var(--accent)]/15 text-[var(--accent)]' : 'bg-[#2a313b] text-[#c7cdd8]'}`}>
      {text}{end && <span className="opacity-40">_</span>}
    </span>
  )
}

export default function TokenizerLab() {
  const [numMerges, setNumMerges] = useState(30)
  const [testText, setTestText] = useState('unhappiest slowest cats')

  const trained = useMemo(() => trainBPE(SAMPLE_CORPUS, 120), [])
  const merges = trained.merges.slice(0, numMerges)
  const words = trained.snapshots[Math.min(numMerges, trained.snapshots.length - 1)]
  const encoded = useMemo(() => encodeBPE(testText, merges), [testText, merges])

  const vocab = new Set<string>()
  for (const w of words) for (const t of w) vocab.add(t)
  const totalTokens = words.reduce((s, w) => s + w.length, 0)
  const testTokens = encoded.reduce((s, w) => s + w.length, 0)

  return (
    <Lab title="Byte pair encoding, one merge at a time"
      takeaway={<>Nothing here knows what a word is. The algorithm only counts which two neighbours
        appear together most often and glues them. Common words end up as one token, rare ones get
        chopped into pieces. That is why a model can spell "the" perfectly and falls apart on a long
        chemical name — and why counting letters is genuinely hard for it, since it never sees them.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <Slider label="merges learned" min={0} max={trained.merges.length} value={numMerges}
                    onChange={setNumMerges} />
          </Panel>
          <Stat label="vocabulary size" value={String(vocab.size)} sub="unique tokens" />
          <Stat label="tokens in corpus" value={String(totalTokens)}
                sub={`${(totalTokens / words.length).toFixed(2)} per word`} tone="good" />
          <Panel title="Try your own text">
            <input value={testText} onChange={e => setTestText(e.target.value)}
              className="w-full rounded-md border border-[var(--line)] bg-[var(--panel-2)] px-2.5 py-1.5 font-mono text-[12px] text-[var(--text)] outline-none focus:border-[var(--accent)]" />
            <div className="mt-2.5 leading-[2]">
              {encoded.map((w, i) => <span key={i} className="mr-2 inline-block">{w.map((t, j) => <Tok key={j} t={t} />)}</span>)}
            </div>
            <div className="mt-2 text-[11px] text-[var(--dim)]">{testTokens} tokens</div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title={`The corpus after ${numMerges} merges`}>
            <div className="max-h-[220px] overflow-y-auto leading-[2.1]">
              {words.slice(0, 48).map((w, i) => (
                <span key={i} className="mr-2.5 inline-block">{w.map((t, j) => <Tok key={j} t={t} />)}</span>
              ))}
            </div>
          </Panel>

          <Panel title="Merges, in the order they were learned">
            <div className="max-h-[200px] overflow-y-auto">
              <table className="w-full font-mono text-[12px]">
                <thead><tr className="text-[var(--dim)]">
                  <th className="py-1 text-left font-normal">#</th>
                  <th className="py-1 text-left font-normal">pair</th>
                  <th className="py-1 text-left font-normal">becomes</th>
                  <th className="py-1 text-right font-normal">seen</th>
                </tr></thead>
                <tbody>
                  {merges.slice().reverse().slice(0, 40).map(m => (
                    <tr key={m.rank} className="border-t border-[var(--line)]">
                      <td className="py-1 text-[#5b6472]">{m.rank}</td>
                      <td className="py-1 text-[#c7cdd8]">
                        {m.a.replace('</w>', '_')} + {m.b.replace('</w>', '_')}
                      </td>
                      <td className="py-1 text-[var(--accent)]">{(m.a + m.b).replace('</w>', '_')}</td>
                      <td className="py-1 text-right text-[var(--dim)]">{m.count}</td>
                    </tr>
                  ))}
                  {merges.length === 0 && (
                    <tr><td colSpan={4} className="py-3 text-[var(--dim)]">No merges yet — every token is a single character.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
