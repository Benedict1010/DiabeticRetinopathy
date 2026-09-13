import { createContext, useContext, useMemo, useState } from 'react'

const AnalysisContext = createContext(null)

const initialClinicalData = {
  patientId: '', age: '', gender: '', ethnicity: '', diabetesDuration: '',
  hba1c: '', bloodGlucose: '', bmi: '', systolic: '', diastolic: '',
}

export function AnalysisProvider({ children }) {
  const [clinicalData, setClinicalData] = useState(initialClinicalData)
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState('')
  const [analysisStatus, setAnalysisStatus] = useState('idle')
  const [result, setResult] = useState(null)

  const value = useMemo(() => ({
    clinicalData, setClinicalData, imageFile, imagePreview, analysisStatus, result,
    setImage(file) {
      setImageFile(file)
      setImagePreview(file ? URL.createObjectURL(file) : '')
    },
    removeImage() { setImageFile(null); setImagePreview('') },
    setAnalysisStatus, setResult,
    resetAnalysis() { setClinicalData(initialClinicalData); setImageFile(null); setImagePreview(''); setAnalysisStatus('idle'); setResult(null) },
  }), [clinicalData, imageFile, imagePreview, analysisStatus, result])
  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>
}

export function useAnalysis() {
  const context = useContext(AnalysisContext)
  if (!context) throw new Error('useAnalysis must be used inside AnalysisProvider')
  return context
}
