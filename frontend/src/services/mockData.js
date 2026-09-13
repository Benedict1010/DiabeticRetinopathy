export const severityLabels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

export const mockPrediction = {
  severity: 2,
  label: 'Moderate diabetic retinopathy',
  shortLabel: 'Moderate DR',
  confidence: 87,
  probabilities: [3, 6, 87, 3, 1],
  observations: [
    'Mock screening output indicates features consistent with the selected severity category.',
    'Image quality is sufficient for this prototype demonstration.',
    'Clinical values are displayed for review only and are not used by a live model.',
  ],
  action: 'Arrange review by a qualified eye-care professional. This prototype output is not a diagnosis.',
  model: 'Demo workflow · no live inference engine connected',
}

export const historyRecords = [
  { id: 'DEMO-1024', date: 'Sep 12, 2026', patient: 'MRN-01842', result: 'Moderate DR', severity: 2, confidence: 87 },
  { id: 'DEMO-1017', date: 'Sep 10, 2026', patient: 'MRN-01607', result: 'No DR', severity: 0, confidence: 94 },
  { id: 'DEMO-1009', date: 'Sep 08, 2026', patient: 'MRN-01389', result: 'Mild DR', severity: 1, confidence: 78 },
  { id: 'DEMO-1003', date: 'Sep 05, 2026', patient: 'MRN-01155', result: 'Severe DR', severity: 3, confidence: 82 },
]
