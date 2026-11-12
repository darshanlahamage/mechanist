import type { ReactNode } from 'react'

export type LensId = 'intuition' | 'mechanism' | 'math' | 'code' | 'systems' | 'failure'

/** The order a lesson reads in. One page, top to bottom — never tabs. */
export const LENSES: { id: LensId; label: string; question: string }[] = [
  { id: 'intuition', label: 'The idea',             question: '' },
  { id: 'mechanism', label: 'Watch it happen',      question: '' },
  { id: 'math',      label: 'The maths behind it',  question: '' },
  { id: 'code',      label: 'Writing it yourself',  question: '' },
  { id: 'systems',   label: 'What it costs to run', question: '' },
  { id: 'failure',   label: 'When it breaks',       question: '' },
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
  hasLab?: boolean          // has a running animation or visual
  hero?: () => ReactNode    // the animation that opens the lesson
  lenses?: Partial<Record<LensId, () => ReactNode>>
}

export interface Track {
  id: string
  title: string
  blurb: string
  concepts: Concept[]
}
