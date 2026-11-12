import type { ReactNode } from 'react'

export type LensId = 'intuition' | 'mechanism' | 'math' | 'code' | 'systems' | 'failure'

export const LENSES: { id: LensId; label: string; question: string }[] = [
  { id: 'intuition', label: 'Intuition',  question: 'What is the idea?' },
  { id: 'mechanism', label: 'Mechanism',  question: 'What happens to the numbers?' },
  { id: 'math',      label: 'Math',       question: 'Why is the formula that shape?' },
  { id: 'code',      label: 'Code',       question: 'How do I write it?' },
  { id: 'systems',   label: 'Systems',    question: 'What does it cost to run?' },
  { id: 'failure',   label: 'Failure',    question: 'How does it break?' },
]

export interface Paper {
  title: string
  year: number
  url: string
  idea: string      // the one thing this paper added
}

export interface Concept {
  id: string
  title: string
  tagline: string
  track: string
  difficulty: 1 | 2 | 3 | 4 | 5
  status: 'planned' | 'draft' | 'stable'
  prereqs?: string[]
  papers?: Paper[]
  hasLab?: boolean
  lenses?: Partial<Record<LensId, () => ReactNode>>
}

export interface Track {
  id: string
  title: string
  blurb: string
  concepts: Concept[]
}
