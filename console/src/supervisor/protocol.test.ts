import { describe, expect, it } from 'vitest'
import {
  EMPTY,
  applyMessage,
  callToShow,
  needsOfficer,
  waitingCalls,
  type DialogueEvent,
  type LocalAction,
  type ServerMessage,
} from './protocol'

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

function run(...messages: (ServerMessage | LocalAction)[]) {
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

  it("labels a line the officer's when the agent's voice says what they sent", () => {
    const state = run(
      started,
      dialogue({ state: 'handed_off', reason: 'taken_over', officer: true }),
      { type: 'officer_said', call_id: 'c1', text: '  ನಾನು   ಅಧಿಕಾರಿ ' },
      { type: 'call_event', call_id: 'c1', event: { type: 'agent_text', text: 'ನಾನು ಅಧಿಕಾರಿ' } },
    )

    expect(state.calls.c1.lines.at(-1)).toEqual({ who: 'officer', text: 'ನಾನು ಅಧಿಕಾರಿ' })
    expect(state.calls.c1.officerQueue).toEqual([])
  })

  it("keeps the agent's own queued lines the agent's after a take-over", () => {
    // The hand-off line is spoken after the officer has already taken the call.
    const state = run(
      started,
      dialogue({ state: 'handed_off', reason: 'safety', officer: true }),
      { type: 'call_event', call_id: 'c1', event: { type: 'agent_text', text: 'ದಯವಿಟ್ಟು ದೂರ ಇರಿ' } },
    )

    expect(state.calls.c1.lines.at(-1)?.who).toBe('agent')
  })

  it("keeps the officer's queued lines theirs after a hand-back", () => {
    const state = run(
      started,
      dialogue({ state: 'handed_off', officer: true }),
      { type: 'officer_said', call_id: 'c1', text: 'ಬರ್ತಿದ್ದೀವಿ' },
      dialogue({ state: 'listening', officer: false, keys: ['hand_back'] }),
      { type: 'call_event', call_id: 'c1', event: { type: 'agent_text', text: 'ಬರ್ತಿದ್ದೀವಿ' } },
    )

    expect(state.calls.c1.lines.at(-1)?.who).toBe('officer')
  })

  it('forgets a line the backend refused to say', () => {
    const state = run(
      started,
      { type: 'officer_said', call_id: 'c1', text: 'ಸರಿ' },
      { type: 'officer_unsaid', call_id: 'c1', text: 'ಸರಿ' },
      { type: 'call_event', call_id: 'c1', event: { type: 'agent_text', text: 'ಸರಿ' } },
    )

    expect(state.calls.c1.lines.at(-1)?.who).toBe('agent')
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

  it('marks calls the console joined part-way through', () => {
    const joined = run({ type: 'calls', thresholds, calls: [{ call_id: 'c0', started_at: 'x', state: 'listening', reason: null, officer: false, slots }] })
    const fresh = run(started)

    expect(joined.calls.c0.joinedMidCall).toBe(true)
    expect(fresh.calls.c1.joinedMidCall).toBe(false)
  })
})

describe('callToShow', () => {
  const at = (id: string, minute: number): ServerMessage => ({
    ...(started as Extract<ServerMessage, { type: 'call_started' }>),
    call_id: id,
    started_at: `2026-10-29T10:0${minute}:00Z`,
  })
  const handOff = (id: string): ServerMessage => ({
    type: 'call_event',
    call_id: id,
    event: { ...(dialogue({ state: 'handed_off', reason: 'safety' }) as { event: DialogueEvent }).event },
  })

  it('opens on the longest-waiting call, then the newest', () => {
    const state = run(at('a', 1), at('b', 2), at('c', 3), handOff('a'), handOff('b'))

    expect(waitingCalls(state).map((c) => c.id)).toEqual(['a', 'b'])
    expect(callToShow(state, null)).toBe('a')
    expect(callToShow(run(at('a', 1), at('b', 2)), null)).toBe('b')
  })

  it("stays on the officer's call when it is taken over or a new call starts", () => {
    const state = run(at('a', 1), handOff('a'))
    const pinned = callToShow(state, null)
    const later = run(at('a', 1), handOff('a'), at('b', 2), at('c', 3))

    expect(callToShow(later, pinned)).toBe('a')
  })
})
