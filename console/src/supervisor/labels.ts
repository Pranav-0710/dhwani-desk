/**
 * Names the console shows for ids in calls and tickets. They mirror data/wards.json and
 * data/decision_schema.json, which labels.test.ts checks.
 */

export const WARDS: Record<string, { name: string; nameEn: string }> = {
  w01: { name: 'ಹೂವಿನಕೇರಿ', nameEn: 'Hoovinakeri' },
  w02: { name: 'ಕೆರೆಕೋಡಿ', nameEn: 'Kerekodi' },
  w03: { name: 'ಸಂತೆಪೇಟೆ', nameEn: 'Santhepete' },
  w04: { name: 'ಗಾಂಧಿನಗರ', nameEn: 'Gandhinagar' },
  w05: { name: 'ಆಲದಮರಹಳ್ಳಿ', nameEn: 'Aaladamarahalli' },
  w06: { name: 'ರೈಲ್ವೆ ಕಾಲೋನಿ', nameEn: 'Railway Colony' },
  w07: { name: 'ತೋಟದಪಾಳ್ಯ', nameEn: 'Thotadapalya' },
  w08: { name: 'ಹೊಸಬಡಾವಣೆ', nameEn: 'Hosabadavane' },
  w09: { name: 'ಕುಂಬಾರಗೇರಿ', nameEn: 'Kumbaragere' },
  w10: { name: 'ಮಾರುತಿನಗರ', nameEn: 'Marutinagar' },
}

export const DEPARTMENTS: Record<string, string> = {
  water: 'Water',
  'sewage-drainage': 'Sewage & drainage',
  'solid-waste': 'Solid waste',
  roads: 'Roads',
  streetlights: 'Streetlights',
  'stray-animals': 'Stray animals',
  other: 'Other',
}

export const URGENCY_LEVELS = ['routine', 'soon', 'urgent', 'emergency'] as const

export function wardLabel(id: string | null | undefined): string {
  if (!id || id === 'unknown') return 'Unknown ward'
  const ward = WARDS[id]
  return ward ? `${ward.nameEn} (${id})` : id
}

export function departmentLabel(id: string | null | undefined): string {
  return id ? (DEPARTMENTS[id] ?? id) : 'Not yet known'
}
