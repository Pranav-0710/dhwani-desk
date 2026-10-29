/**
 * The supervisor console's view of `/ws/console` (backend/README.md, "Supervisor console"), and
 * a pure reducer from the server's messages to the state the page shows.
 */

export type CallState = 'listening' | 'handed_off' | 'closed'

export interface Slots {
  department: string | null
  ward: string | null
  location_given: boolean
}

/** Every answer behind a reply: yes/no questions as the probability of "yes". */
export interface Decision {
  department: string
  department_confidence: number
  ward: string
  urgency: string
  safety_hazard: number
  has_location: number
  repeat_complaint: number
  wants_human: number
  caller_state: string
  out_of_scope: number
  contains_personal_id: number
}

/** The safety step's answers, when a call was handed off before the rest was decided. */
export interface Screening {
  safety_hazard: number
  urgency: string
  wants_human: number
  out_of_scope: number
}

export interface Ticket {
  number: string
  department: string
  ward: string
  urgency: string
  repeat_complaint: boolean
}

export interface Thresholds {
  safety_hazard: number
  emergency_urgency: string
  wants_human: number
  out_of_scope: number
  has_location: number
  repeat_complaint: number
  department_confidence: number
  hand_off_caller_states: string[]
  max_unrouted_turns: number
  max_location_asks: number
}

interface CallSummary {
  call_id: string
  started_at: string
  state: CallState
  reason: string | null
  officer: boolean
  slots: Slots
  turns?: string[]
}

export interface DialogueEvent {
  type: 'dialogue'
  keys: string[]
  state: CallState
  reason: string | null
  officer: boolean
  slots: Slots
  ticket: Ticket | null
  decision: Decision | null
  screening: Screening | null
}

/** What one call's browser got, as the console sees it in `call_event`; others are ignored. */
export type CallEvent =
  | { type: 'transcript'; text: string }
  | { type: 'agent_text'; text: string }
  | DialogueEvent
  | { type: 'other' }

export type ServerMessage =
  | { type: 'calls'; calls: CallSummary[]; thresholds: Thresholds }
  | ({ type: 'call_started' } & CallSummary)
  | { type: 'call_event'; call_id: string; event: CallEvent }
  | { type: 'call_closed'; call_id: string }
  | { type: 'done'; command: string; call_id: string; id?: string | number }
  | { type: 'error'; message: string; command?: string; call_id?: string; id?: string | number }

/** Actions from the page itself: the officer sent a line (or it failed), so the next time the
 * agent's voice says those words they are the officer's. */
export type LocalAction =
  | { type: 'officer_said'; call_id: string; text: string }
  | { type: 'officer_unsaid'; call_id: string; text: string }

export type Speaker = 'caller' | 'agent' | 'officer'

export interface Line {
  who: Speaker
  text: string
}

export interface LiveCall {
  id: string
  startedAt: string
  state: CallState
  reason: string | null
  officer: boolean
  slots: Slots
  lines: Line[]
  decision: Decision | null
  /** The safety step's answers while the full decision for this turn is still to come. */
  screening: Screening | null
  ticket: Ticket | null
  /** The call's socket has closed; kept on screen so its last state can be read. */
  ended: boolean
  /** Lines the officer sent that the agent's voice has not said yet, oldest first. Lines are
   * spoken some time after they are queued, so `officer` at that moment cannot tell. */
  officerQueue: string[]
  /** The console joined after the call started: decisions before then were not seen. */
  joinedMidCall: boolean
}

export interface ConsoleState {
  calls: Record<string, LiveCall>
  /** Call ids, newest first. */
  order: string[]
  thresholds: Thresholds | null
}

export const EMPTY: ConsoleState = { calls: {}, order: [], thresholds: null }

/** An officer's line as the backend speaks it (Dialogue.officer_says). */
export function spokenForm(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(' ')
}

function fromSummary(summary: CallSummary, joinedMidCall: boolean): LiveCall {
  return {
    id: summary.call_id,
    startedAt: summary.started_at,
    state: summary.state,
    reason: summary.reason,
    officer: summary.officer,
    slots: summary.slots,
    lines: (summary.turns ?? []).map((text) => ({ who: 'caller', text })),
    decision: null,
    screening: null,
    ticket: null,
    ended: false,
    officerQueue: [],
    joinedMidCall,
  }
}

function withCall(state: ConsoleState, call: LiveCall): ConsoleState {
  const order = state.order.includes(call.id) ? state.order : [call.id, ...state.order]
  return { ...state, calls: { ...state.calls, [call.id]: call }, order }
}

function applyEvent(call: LiveCall, event: CallEvent): LiveCall {
  switch (event.type) {
    case 'transcript':
      return event.text.trim()
        ? { ...call, lines: [...call.lines, { who: 'caller', text: event.text }] }
        : call
    case 'agent_text': {
      const [next, ...rest] = call.officerQueue
      const officers = next !== undefined && spokenForm(event.text) === next
      const who: Speaker = officers ? 'officer' : 'agent'
      return {
        ...call,
        lines: [...call.lines, { who, text: event.text }],
        officerQueue: officers ? rest : call.officerQueue,
      }
    }
    case 'dialogue':
      return {
        ...call,
        state: event.state,
        reason: event.reason,
        officer: event.officer,
        slots: event.slots,
        ticket: event.ticket ?? call.ticket,
        // A full decision replaces the safety step's answers; a screening alone means the
        // rest of the decision is still running for this turn.
        decision: event.decision ?? (event.screening ? null : call.decision),
        screening: event.decision ? null : (event.screening ?? call.screening),
      }
    default:
      return call
  }
}

export function applyMessage(state: ConsoleState, message: ServerMessage | LocalAction): ConsoleState {
  switch (message.type) {
    case 'officer_said':
    case 'officer_unsaid': {
      const call = state.calls[message.call_id]
      if (!call) return state
      const line = spokenForm(message.text)
      const queue = [...call.officerQueue]
      if (message.type === 'officer_said') queue.push(line)
      else if (queue.includes(line)) queue.splice(queue.indexOf(line), 1)
      return withCall(state, { ...call, officerQueue: queue })
    }
    case 'calls': {
      const calls = Object.fromEntries(message.calls.map((c) => [c.call_id, fromSummary(c, true)]))
      const order = [...message.calls]
        .sort((a, b) => b.started_at.localeCompare(a.started_at))
        .map((c) => c.call_id)
      return { calls, order, thresholds: message.thresholds }
    }
    case 'call_started':
      return withCall(state, fromSummary(message, false))
    case 'call_event': {
      const call = state.calls[message.call_id]
      return call ? withCall(state, applyEvent(call, message.event)) : state
    }
    case 'call_closed': {
      const call = state.calls[message.call_id]
      return call ? withCall(state, { ...call, ended: true }) : state
    }
    default:
      return state
  }
}

/** The call is waiting for an officer: handed off, and nobody has taken it yet. */
export function needsOfficer(call: LiveCall): boolean {
  return !call.ended && call.state === 'handed_off' && !call.officer
}

/** Calls waiting for an officer, the longest-waiting first. */
export function waitingCalls(state: ConsoleState): LiveCall[] {
  return state.order
    .map((id) => state.calls[id])
    .filter(needsOfficer)
    .reverse()
}

/** Which call the page shows: the officer's choice, else the longest-waiting, else the newest.
 * The page pins whatever this returns, so it never moves on by itself. */
export function callToShow(state: ConsoleState, chosen: string | null): string | null {
  if (chosen && state.calls[chosen]) return chosen
  return waitingCalls(state)[0]?.id ?? state.order[0] ?? null
}

/** Why a call was handed off or closed, for an officer. */
export const REASON_LABEL: Record<string, string> = {
  safety: 'Safety risk',
  wants_human: 'Caller asked for an officer',
  distressed: 'Caller is distressed',
  abusive: 'Caller is abusive',
  unroutable: "Couldn't tell the department",
  no_location: "Couldn't get a location",
  decider_error: 'The decider failed',
  taken_over: 'Taken over by an officer',
  out_of_scope: 'Not a civic matter: redirected',
  ticket: 'Ticket filed',
}
