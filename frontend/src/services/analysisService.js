import { mockPrediction } from './mockData'

export async function submitAnalysis({ clinicalData, imageFile }) {
  // Stable future integration seam. This deliberate delay simulates a demo workflow only.
  await new Promise((resolve) => setTimeout(resolve, 1100))
  return { ...mockPrediction, generatedAt: new Date().toISOString(), clinicalData, fileName: imageFile?.name }
}
