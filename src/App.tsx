import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './components/Home'
import TrackPage from './components/TrackPage'
import ConceptPage from './components/ConceptPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/track/:id" element={<TrackPage />} />
        <Route path="/c/:id" element={<ConceptPage />} />
        <Route path="*" element={<Home />} />
      </Routes>
    </Layout>
  )
}
