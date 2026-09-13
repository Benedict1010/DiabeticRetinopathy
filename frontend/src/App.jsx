import { Navigate, Route, Routes } from 'react-router-dom'
import Sidebar from './components/layout/Sidebar'
import Header from './components/layout/Header'
import Dashboard from './pages/Dashboard'
import ClinicalData from './pages/ClinicalData'
import ImageUpload from './pages/ImageUpload'
import ReviewAnalysis from './pages/ReviewAnalysis'
import AnalysisLoading from './pages/AnalysisLoading'
import Results from './pages/Results'
import History from './pages/History'
import About from './pages/About'

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="app-content">
        <Header />
        <main className="page-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/clinical-data" element={<ClinicalData />} />
            <Route path="/image-upload" element={<ImageUpload />} />
            <Route path="/review" element={<ReviewAnalysis />} />
            <Route path="/analysis" element={<AnalysisLoading />} />
            <Route path="/results" element={<Results />} />
            <Route path="/history" element={<History />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
