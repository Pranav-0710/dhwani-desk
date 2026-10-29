import { describe, expect, it } from 'vitest'
import { EMPTY, applyMessage, needsOfficer, type DialogueEvent, type ServerMessage } from './protocol'

const thresholds = {
  safety_hazard: 0.6,
  emergency_urgency: 'emergency',
  wants_human: 0.7,
  out_of_scope: 0.7,
  has_location: 0.5,
  repeat_complaint: 0.5,
  department_confidence: 0.6,
  hand_off_caller_states: ['distressed', 'abusive'],
  max_unrouted_turns: 2,
  max_location_asks: 2,
}
const slots = { department: null, ward: null, location_given: false }

function dialogue(overrides: Partial<DialogueEvent>): ServerMessage {
  const event: DialogueEvent = {
    type: 'dialogue',
    keys: [],
    state: 'listening',
    reason: null,
    officer: false,
    slots,
    ticket: null,
    decision: null,
    screening: null,
    ...overrides,
  }
  return { type: 'call_event', call_id: 'c1', event }
}

const decision = {
  department: 'roads',
  department_confidence: 0.82,
  ward: 'w04',
  urgency: 'urgent',
  safety_hazard: 0.1,
  has_location: 0.9,
  repeat_complaint: 0.2,
  wants_human: 0.05,
  caller_state: 'calm',
  out_of_scope: 0.02,
  contains_personal_id: 0.01,
}

function run(...messages: ServerMessage[]) {
  return messages.reduce(applyMessage, EMPTY)
}

const started: ServerMessage = {
  type: 'call_started',
  call_id: 'c1',
  started_at: '2026-10-29T10:00:00Z',
  state: 'listening',
  reason: null,
  officer: false,
  slots,
}

describe('applyMessage', () => {
  it('lists the calls a console joins mid-way, newest first, with what was said so far', () => {
    const summary = { state: 'listening' as const, reason: null, officer: false, slots }
    const state = run({
      type: 'calls',
      thresholds,
      calls: [
        { ...summary, call_id: 'old', started_at: '2026-10-29T09:00:00Z', turns: ['ರಸ್ತೆ ಹಾಳಾಗಿದೆ'] },
        { ...summary, call_id: 'new', started_at: '2026-10-29T10:00:00Z' },
      ],
    })

    expect(state.order).toEqual(['new', 'old'])
    expect(state.calls.old.lines).toEqual([{ who: 'caller', text: 'ರಸ್ತೆ ಹಾಳಾಗಿದೆ' }])
    expect(state.thresholds).toEqual(thresholds)
  })

  it('builds the transcript from caller segments and agent lines', () => {
    const state = run(
      started,
      { type: 'call_event', call_id: 'c1', event: { type: 'transcript', text: 'ಗುಂಡಿ ಬಿದ್ದಿದೆ' } },
      { type: 'call_event', call_id: 'c1', event: { type: 'transcript', text: '  ' } },
      { type: 'call_event', call_id: 'c1', event: { type: 'agent_text', text: 'ಎಲ್ಲಿ?' } },
    )

    expect(state.calls.c1.lines).toEqual([
      { who: 'caller', text: 'ಗುಂಡಿ ಬಿದ್ದಿದೆ' },
      { who: 'agent', text: 'ಎಲ್ಲಿ?' },
    ])
  })

  it("marks what the agent's voice says for an officer as the officer's", () => {
    const state = run(
      started,
      dialogue({ state: 'handed_off', reason: 'taken_over', officer: true }),
      { type: 'call_event', call_id: 'c1', event: { type: 'agent_text', text: 'ನಾನು ಅಧಿಕಾರಿ' } },
    )

    expect(state.calls.c1.lines.at(-1)).toEqual({ who: 'officer', text: 'ನಾನು ಅಧಿಕಾರಿ' })
  })

  it('shows the safety step until the full decision replaces it', () => {
    const screening = { safety_hazard: 0.93, urgency: 'emergency', wants_human: 0.1, out_of_scope: 0.05 }
    const screened = run(started, dialogue({ state: 'handed_off', reason: 'safety', screening }))

    expect(screened.calls.c1.screening).toEqual(screening)
    expect(screened.calls.c1.decision).toBeNull()
    expect(needsOfficer(screened.calls.c1)).toBe(true)

    const decided = applyMessage(screened, dialogue({ state: 'handed_off', reason: 'safety', decision }))
    expect(decided.calls.c1.decision).toEqual(decision)
    expect(decided.calls.c1.screening).toBeNull()
  })

  it('keeps the last decision across a reply that carries none', () => {
    const state = run(started, dialogue({ decision }), dialogue({ keys: ['hand_back'] }))

    expect(state.calls.c1.decision).toEqual(decision)
  })

  it('stops asking for an officer once one has the call, and when the call ends', () => {
    const handed = run(started, dialogue({ state: 'handed_off', reason: 'wants_human' }))
    expect(needsOfficer(handed.calls.c1)).toBe(true)

    const taken = applyMessage(handed, dialogue({ state: 'handed_off', reason: 'wants_human', officer: true }))
    expect(needsOfficer(taken.calls.c1)).toBe(false)

    const closed = applyMessage(handed, { type: 'call_closed', call_id: 'c1' })
    expect(closed.calls.c1.ended).toBe(true)
    expect(needsOfficer(closed.calls.c1)).toBe(false)
  })

  it('ignores events for a call it never heard start', () => {
    const state = run({ type: 'call_event', call_id: 'ghost', event: { type: 'transcript', text: 'x' } })

    expect(state).toEqual(EMPTY)
  })
})
