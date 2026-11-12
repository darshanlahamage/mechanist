import TokenizerLab from '../../labs/TokenizerLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const tokenization: Concept = {
  id: 'tokenization',
  title: 'Tokenisation',
  tagline: 'The model never sees letters. It sees chunks that a counting algorithm decided were common.',
  track: 'transformer',
  difficulty: 2,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'Neural Machine Translation of Rare Words with Subword Units', year: 2015,
      url: 'https://arxiv.org/abs/1508.07909',
      idea: 'Borrow a compression algorithm. Merge the most frequent adjacent pair, repeat, and rare words fall apart into pieces you have seen.' },
    { title: 'SentencePiece', year: 2018, url: 'https://arxiv.org/abs/1808.06226',
      idea: 'Treat the raw text as a stream of bytes with no pre-splitting on spaces, so the same tokenizer works for any language.' },
    { title: 'Byte Latent Transformer', year: 2024, url: 'https://arxiv.org/abs/2412.09871',
      idea: 'Drop the fixed vocabulary. Group bytes into patches based on how predictable they are.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>A model works with a fixed list of symbols. You have to turn text into numbers from that
        list before anything else can happen.</p>
        <p>One symbol per letter is too few — sequences get very long and the model wastes its
        capacity learning to spell. One symbol per word is too many — you would need millions of
        them, and you would still meet words you have never seen.</p>
        <p>Byte pair encoding splits the difference. Start with characters. Count which two
        neighbours appear together most often. Glue that pair into one new symbol. Repeat a few
        thousand times. Common words end up as single tokens, rare ones get built out of pieces.</p>
        <Note kind="key" title="Nothing here understands language">
          <p>The algorithm has no idea what a word or a prefix is. It is counting pairs. The fact
          that it discovers "ing" and "un" is a side effect of those letters being common together.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Drag the merge count from zero upward and watch the vocabulary build itself:</p>
        <TokenizerLab />
        <h3>Training, step by step</h3>
        <ul>
          <li>Split the corpus into words, then into characters, with a marker for word end.</li>
          <li>Count every adjacent pair across the whole corpus.</li>
          <li>Take the most frequent pair and replace every occurrence with a single new symbol.</li>
          <li>Record that merge and its rank. Repeat until you hit your target vocabulary size.</li>
        </ul>
        <h3>Encoding new text</h3>
        <p>Split into characters, then apply the learned merges <b>in the order they were learned</b>.
        Rank order matters: a later merge can only fire on symbols that earlier merges produced.
        That is why the merge list has to ship with the model.</p>
      </>
    ),
    math: () => (
      <>
        <h3>The tradeoff you are choosing</h3>
        <p>A bigger vocabulary means fewer tokens per piece of text, so shorter sequences and less
        compute per document. It also means a bigger embedding table and a bigger output softmax.</p>
        <Eq block tex="\text{embedding + head params} = 2 \cdot V \cdot d_{model}" />
        <p>With <Eq tex="V = 128{,}000" /> and <Eq tex="d_{model} = 4096" /> that is about a billion
        parameters spent on the vocabulary alone. Attention cost falls with shorter sequences as
        <Eq tex="n^2" />, so there is a real optimum, and it has moved upward over time as models
        got bigger and context got longer.</p>
        <h3>Fertility</h3>
        <p>Tokens per word is called fertility. English sits near 1.3 for most modern tokenizers.
        Languages written in other scripts, or with rich morphology, often sit at 2 to 4 — which
        means the same sentence costs two to four times more to process and to buy.</p>
        <Eq block tex="\text{fertility} = \frac{\text{tokens}}{\text{words}}" />
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`from collections import Counter

def train_bpe(corpus, num_merges):
    words = [list(w) + ['</w>'] for w in corpus.lower().split()]
    merges = []

    for _ in range(num_merges):
        pairs = Counter()
        for w in words:
            for i in range(len(w) - 1):
                pairs[(w[i], w[i + 1])] += 1
        if not pairs:
            break

        (a, b), count = pairs.most_common(1)[0]
        if count < 2:
            break

        new_words = []
        for w in words:
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(w[i]); i += 1
            new_words.append(out)
        words = new_words
        merges.append((a, b))

    return merges`}</Code>
        <Code lang="python">{`from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# always look at what your text actually becomes
print(tok.tokenize("strawberry"))        # the famous letter-counting problem
print(tok.tokenize("1234567890"))        # numbers split in ways that break arithmetic
print(tok.tokenize("    indented"))      # whitespace is part of the token

# cost in a non-English script
for text in ["hello world", "नमस्ते दुनिया", "こんにちは世界"]:
    print(len(tok.encode(text)), text)`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>Tokenisation decides what you pay</h3>
        <p>Every API bills by the token, every context window is measured in tokens, and attention
        cost grows with the square of them. A tokenizer that splits a language badly makes that
        language more expensive to use, by a factor of two or three, permanently.</p>
        <h3>The embedding table is not free</h3>
        <p>Large vocabularies have to be split across GPUs during training (vocabulary parallelism)
        because the output logits matrix is <Eq tex="\text{batch} \times \text{seq} \times V" /> and
        at V = 128k that is one of the biggest tensors in the whole forward pass.</p>
        <h3>You cannot change it later</h3>
        <p>The tokenizer is fixed before pretraining starts. Everything the model knows is expressed
        in those symbols. Swapping it afterwards means retraining the embedding table and the output
        head, and usually most of the model with them.</p>
      </>
    ),
    failure: () => (
      <>
        <h3>Things tokenisation causes that look like model failures</h3>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Cannot count letters in a word', 'It never sees letters. "strawberry" may be two or three tokens, and the letters inside them are invisible.'],
              ['Arithmetic errors on long numbers', 'Digits group inconsistently. 1234 might be one token while 1235 is two.'],
              ['Worse quality in some languages', 'Higher fertility means fewer real words fit in context, and less of the training budget went there.'],
              ['Code formatting breaks', 'Indentation is encoded in tokens. Mixed tabs and spaces produce tokens the model rarely saw.'],
              ['Strange outputs from rare strings', 'Glitch tokens — symbols that exist in the vocabulary but almost never appeared in training, so their embeddings are untrained.'],
              ['Prompt works, same prompt with trailing space fails', 'A trailing space changes the tokenisation of the next word entirely.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Before blaming the model">
          <p>Print the tokens. Most "the model is bad at X" reports about spelling, counting,
          formatting or a specific language turn out to be the tokenizer, and no amount of prompting
          fixes it.</p>
        </Note>
      </>
    ),
  },
}
