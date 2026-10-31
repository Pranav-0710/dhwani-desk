import { describe, expect, it } from 'vitest'
import schemaJson from '../../../data/decision_schema.json?raw'
import wardsJson from '../../../data/wards.json?raw'
import { DEPARTMENTS, URGENCY_LEVELS, WARDS, wardLabel } from './labels'

describe('labels', () => {
  it('names every ward in data/wards.json, and no others', () => {
    const wards = JSON.parse(wardsJson).wards as { id: string; name: string; name_en: string }[]

    expect(Object.keys(WARDS)).toEqual(wards.map((w) => w.id))
    for (const ward of wards) expect(WARDS[ward.id]).toEqual({ name: ward.name, nameEn: ward.name_en })
  })

  it('names every department and urgency level in the decision schema', () => {
    const questions = JSON.parse(schemaJson).questions

    expect(Object.keys(DEPARTMENTS)).toEqual(Object.keys(questions.department.criteria))
    expect([...URGENCY_LEVELS]).toEqual(questions.urgency.criteria)
  })

  it('says when the ward is not known', () => {
    expect(wardLabel('unknown')).toBe('Unknown ward')
    expect(wardLabel(null)).toBe('Unknown ward')
    expect(wardLabel('w04')).toBe('Gandhinagar (w04)')
  })
})
