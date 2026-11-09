import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type RefObject } from 'react'
import { ConsoleClient, type Command, type ConsoleStatus } from './console-client'
import { DEPARTMENTS, URGENCY_LEVELS, WARDS, departmentLabel, wardLabel } from './labels'
import {
  EMPTY,
  REASON_LABEL,
  applyMessage,
  callToShow,
  callsByUrgency,
  isSafety,
  needsOfficer,
  waitingCalls,
  type ConsoleState,
  type LocalAction,
  type Decision,
  type LiveCall,
  type Screening,
  type ServerMessage,
  type Thresholds,
} from './protocol'
import { ALL, NO_FILTER, fetchTickets, filterTickets, type StoredTicket, type TicketFilter } from './tickets'
import './supervisor.css'

const TOKEN_KEY = 'dhwani-console-token'
const TICKETS_REFRESH_MS = 10_000
/** The backend speaks at most this many characters of an officer's line. */
const MAX_OFFICER_LINE = 300

const STATUS_LABEL: Record<ConsoleStatus, string> = {
  connecting: 'Connecting…',
  connected: 'Live',
  refused: 'Token refused',
  closed: 'Disconnected',
}

function readToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

function saveToken(token: string | null): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token)
    else sessionStorage.removeItem(TOKEN_KEY)
  } catch {
    // Private windows can refuse storage; the officer just types the token again.
  }
}

export default function Supervisor() {
  const [token, setToken] = useState(readToken)
  // Bumped to connect again with the same token: a Reconnect, or the same token re-entered.
  const [attempt, setAttempt] = useState(0)
  const [state, dispatch] = useReducer(
    (current: ConsoleState, message: ServerMessage | LocalAction) => applyMessage(current, message),
    EMPTY,
  )
  const [status, setStatus] = useState<ConsoleStatus>('closed')
  const [problem, setProblem] = useState<string | null>(null)
  const [ticketsError, setTicketsError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [tickets, setTickets] = useState<StoredTicket[]>([])
  const client = useRef<ConsoleClient | null>(null)
  const takeOverButton = useRef<HTMLButtonElement>(null)
  // Set by the banner: once its call is on screen, the keyboard goes to that call's Take over.
  const goToTakeOver = useRef(false)

  useEffect(() => {
    if (!goToTakeOver.current) return
    goToTakeOver.current = false
    takeOverButton.current?.focus()
  })

  const refreshTickets = useCallback(() => {
    if (!token) return
    fetchTickets(token)
      .then((list) => {
        setTickets(list)
        setTicketsError(null)
      })
      .catch((error: Error) => setTicketsError(error.message))
  }, [token])
  // The socket's handlers read this, so a new refreshTickets never reconnects the socket.
  const refresh = useRef(refreshTickets)
  useEffect(() => {
    refresh.current = refreshTickets
  }, [refreshTickets])

  useEffect(() => {
    if (!token) return
    const connection = new ConsoleClient(
      (message) => {
        dispatch(message)
        if (message.type === 'call_closed') refresh.current()
      },
      (next, detail) => {
        setStatus(next)
        if (next === 'refused') {
          setProblem(detail ?? 'The console token was refused')
          saveToken(null)
        } else if (next === 'closed') {
          setProblem(detail ? `Disconnected: ${detail}` : 'Disconnected from the backend')
        } else if (next === 'connected') {
          setProblem(null)
        }
      },
    )
    connection.connect(token)
    client.current = connection
    return () => {
      connection.close()
      client.current = null
    }
  }, [token, attempt])

  useEffect(() => {
    if (status !== 'connected') return
    refreshTickets()
    const timer = setInterval(refreshTickets, TICKETS_REFRESH_MS)
    return () => clearInterval(timer)
  }, [status, refreshTickets])

  // The page shows one call and stays on it: chosen by the officer, or else picked once (the
  // longest-waiting, or the newest) and pinned, so a take-over or a new call never moves it.
  const shown = callToShow(state, selected)
  if (shown && shown !== selected) setSelected(shown)
  const calls = callsByUrgency(state)
  const waiting = waitingCalls(state)
  const first = waiting.at(0)
  const current = shown ? state.calls[shown] : null

  if (!token || status === 'refused') {
    return (
      <TokenGate
        problem={status === 'refused' ? problem : null}
        onSubmit={(value) => {
          saveToken(value)
          setStatus('connecting')
          setToken(value)
          setAttempt((n) => n + 1)
        }}
      />
    )
  }

  const sendFor = (callId: string) => async (command: Command, text?: string) => {
    const connection = client.current
    if (!connection) throw new Error('The console is not connected')
    if (command !== 'say' || !text) return connection.send(command, callId, text)
    dispatch({ type: 'officer_said', call_id: callId, text })
    try {
      await connection.send(command, callId, text)
    } catch (failure) {
      dispatch({ type: 'officer_unsaid', call_id: callId, text })
      throw failure
    }
  }

  return (
    <div className="supervisor">
      <header className="sv-header">
        <div>
          <h1>Nagara Seva supervisor console</h1>
          <p className="subtitle">City Corporation of Hosapura · live calls and tickets</p>
        </div>
        <span className={`sv-status sv-status-${status}`} role="status">
          {STATUS_LABEL[status]}
        </span>
      </header>

      <div role="alert" aria-live="assertive">
        {first && (
          <button
            type="button"
            className={`sv-alert${isSafety(first) ? ' is-safety' : ''}`}
            onClick={() => {
              // Already on screen: nothing re-renders, so focus Take over now.
              if (first.id === shown) {
                takeOverButton.current?.focus()
                return
              }
              goToTakeOver.current = true
              setSelected(first.id)
            }}
          >
            <span className="sv-alert-mark" aria-hidden="true">
              !
            </span>
            <span className="sv-alert-text">
              {REASON_LABEL[first.reason ?? ''] ?? first.reason}: {departmentLabel(first.slots.department)},{' '}
              {wardLabel(first.slots.ward)}
              <span className="sv-alert-more">
                Waiting for an officer
                {waiting.length > 1 ? `, and ${waiting.length - 1} more waiting` : ''}
              </span>
            </span>
            <span className="sv-alert-action">{first.id === shown ? 'Go to Take over' : 'Show this call'}</span>
          </button>
        )}
      </div>
      {problem && (
        <p className="error" role="alert">
          {problem}{' '}
          {status === 'closed' && (
            <button type="button" className="sv-secondary sv-inline" onClick={() => setAttempt((n) => n + 1)}>
              Reconnect
            </button>
          )}
        </p>
      )}

      <div className="sv-main">
        <CallList calls={calls} current={shown} onSelect={setSelected} />
        {current ? (
          <CallView
            key={current.id}
            call={current}
            thresholds={state.thresholds}
            send={sendFor(current.id)}
            takeOverButton={takeOverButton}
          />
        ) : (
          <section className="sv-call sv-empty" aria-label="Selected call">
            {status === 'connected' ? (
              <>
                <h2>No live calls</h2>
                <p>
                  A call shows here as soon as it starts. <a href="/">Open the phone page</a> to place one.
                </p>
              </>
            ) : (
              <h2>{status === 'closed' && problem ? 'Not connected, so no calls can show' : 'Connecting…'}</h2>
            )}
          </section>
        )}
      </div>

      <TicketTable tickets={tickets} error={ticketsError} />
    </div>
  )
}

function TokenGate({ problem, onSubmit }: { problem: string | null; onSubmit: (token: string) => void }) {
  const [value, setValue] = useState('')
  return (
    <form
      className="sv-gate"
      onSubmit={(event) => {
        event.preventDefault()
        if (value.trim()) onSubmit(value.trim())
      }}
    >
      <h1>Supervisor console</h1>
      <p className="subtitle">Enter the console token.</p>
      <label>
        Console token
        <input type="password" autoComplete="off" value={value} onChange={(e) => setValue(e.target.value)} />
      </label>
      {problem && (
        <p className="error" role="alert">
          {problem}
        </p>
      )}
      <button className="call" type="submit">
        Open console
      </button>
    </form>
  )
}

function stateLabel(call: LiveCall): string {
  if (call.ended) return 'Ended'
  if (call.officer) return 'Officer on the line'
  if (call.state === 'handed_off') return 'Waiting for an officer'
  if (call.state === 'closed') return 'Closing'
  return 'Agent handling'
}

function CallList({
  calls,
  current,
  onSelect,
}: {
  calls: LiveCall[]
  current: string | null
  onSelect: (id: string) => void
}) {
  return (
    <nav className="sv-calls" aria-label="Live calls">
      <h2>Calls{calls.length > 0 ? ` (${calls.length})` : ''}</h2>
      {calls.length === 0 && <p className="subtitle sv-empty-calls">None right now</p>}
      <ul>
        {calls.map((call) => {
          const waiting = needsOfficer(call)
          const urgency = (call.decision ?? call.screening)?.urgency
          const pressing = urgency === 'urgent' || urgency === 'emergency'
          return (
            <li key={call.id}>
              <button
                type="button"
                className={[
                  'sv-call-item',
                  call.id === current ? 'is-current' : '',
                  waiting ? 'is-waiting' : '',
                  waiting && isSafety(call) ? 'is-safety' : '',
                  call.ended ? 'is-ended' : '',
                ].join(' ')}
                aria-current={call.id === current ? 'true' : undefined}
                onClick={() => onSelect(call.id)}
              >
                {isSafety(call) && !call.ended ? (
                  <span className="sv-chip sv-urgency-emergency">{REASON_LABEL.safety}</span>
                ) : (
                  waiting && <span className="sv-chip sv-urgency-urgent">{REASON_LABEL[call.reason ?? ''] ?? call.reason}</span>
                )}
                <span className="sv-call-dept">{departmentLabel(call.slots.department)}</span>
                <span>{wardLabel(call.slots.ward)}</span>
                <span className="sv-call-state">{stateLabel(call)}</span>
                <span className="sv-call-time">
                  {new Date(call.startedAt).toLocaleTimeString()}
                  {pressing && !call.ended ? ` · ${urgency}` : ''}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

function CallView({
  call,
  thresholds,
  send,
  takeOverButton,
}: {
  call: LiveCall
  thresholds: Thresholds | null
  send: (command: Command, text?: string) => Promise<void>
  takeOverButton: RefObject<HTMLButtonElement | null>
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const transcript = useRef<HTMLOListElement>(null)
  const sayInput = useRef<HTMLInputElement>(null)
  // The button just pressed goes away when the call changes hands: its next control gets the
  // keyboard, whichever comes first of the command's answer and the call's new state.
  const handOver = useRef<'to-officer' | 'to-agent' | null>(null)

  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight })
  }, [call.lines.length])

  useEffect(() => {
    if (handOver.current === 'to-officer' && call.officer) {
      handOver.current = null
      sayInput.current?.focus()
    } else if (handOver.current === 'to-agent' && call.state !== 'handed_off') {
      handOver.current = null
      takeOverButton.current?.focus()
    }
  }, [call.officer, call.state, takeOverButton])

  async function run(command: Command, text?: string) {
    if (busy) return
    setBusy(true)
    setError(null)
    if (command === 'take_over') handOver.current = 'to-officer'
    if (command === 'hand_back') handOver.current = 'to-agent'
    try {
      await send(command, text)
      if (command === 'say') {
        setDraft('')
        sayInput.current?.focus()
      }
    } catch (failure) {
      handOver.current = null
      setError((failure as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const live = !call.ended && call.state !== 'closed'
  const waiting = needsOfficer(call)
  const reason = REASON_LABEL[call.reason ?? ''] ?? call.reason
  const handBack = call.state === 'handed_off' && (
    <button type="button" className="sv-secondary" aria-disabled={busy} onClick={() => void run('hand_back')}>
      Hand back to the agent
    </button>
  )
  return (
    <section className="sv-call" aria-label="Selected call">
      <div className="sv-call-body">
        <div className="sv-call-head">
          <div>
            <h2>{departmentLabel(call.slots.department)}</h2>
            <p className="subtitle">
              {wardLabel(call.slots.ward)} · {stateLabel(call)}
              {reason && !waiting ? ` · ${reason}` : ''}
            </p>
          </div>
          {call.ticket && <span className="sv-ticket-chip">Ticket #{call.ticket.number}</span>}
        </div>

        {!live ? (
          <div className="sv-action">
            <p>{call.ended ? 'This call has ended.' : 'The agent is closing this call.'}</p>
          </div>
        ) : call.officer ? (
          <div className="sv-action is-officer">
            <p>
              <strong>Officer on the line</strong>
              The agent stays quiet and says only what is typed below.
            </p>
            {handBack}
          </div>
        ) : (
          <div className={`sv-action${waiting ? ' is-waiting' : ''}`}>
            <p>
              <strong>{waiting ? (reason ?? 'Handed off') : 'Agent handling'}</strong>
              {waiting ? 'The caller is waiting for an officer.' : 'Take over to speak to the caller yourself.'}
            </p>
            <button
              type="button"
              className="sv-take-over"
              ref={takeOverButton}
              aria-disabled={busy}
              onClick={() => void run('take_over')}
            >
              Take over
            </button>
            {handBack}
          </div>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <ol className="sv-transcript" ref={transcript} aria-label="Transcript" aria-live="polite">
          {call.lines.length === 0 && <li className="sv-transcript-empty">Nothing has been said yet.</li>}
          {call.lines.map((line, index) => (
            <li key={index} className={`sv-line sv-${line.who}`}>
              <span className="who">{line.who === 'officer' ? 'Officer (agent voice)' : line.who === 'caller' ? 'Caller' : 'Agent'}</span>
              <span className="text" lang="kn">
                {line.text}
              </span>
            </li>
          ))}
        </ol>

        {live && call.officer && (
          <form
            className="sv-say"
            onSubmit={(event) => {
              event.preventDefault()
              if (draft.trim()) void run('say', draft.trim())
            }}
          >
            <label className="sr-only" htmlFor={`say-${call.id}`}>
              Say to the caller
            </label>
            <input
              id={`say-${call.id}`}
              ref={sayInput}
              lang="kn"
              value={draft}
              maxLength={MAX_OFFICER_LINE}
              placeholder="Type in Kannada; the agent's voice says it"
              aria-describedby={`say-count-${call.id}`}
              onChange={(e) => setDraft(e.target.value)}
              readOnly={busy}
            />
            <button className="call" type="submit" aria-disabled={busy} disabled={!draft.trim()}>
              Say
            </button>
            <span className="sv-say-count" id={`say-count-${call.id}`}>
              {draft.length} of {MAX_OFFICER_LINE} characters
            </span>
          </form>
        )}
      </div>

      <DecisionBars decision={call.decision} screening={call.screening} thresholds={thresholds} joinedMidCall={call.joinedMidCall} />
    </section>
  )
}

/** `risk`: past its threshold the call is handed off or redirected, so it is shown as an alarm.
 * The others past theirs only mean the agent has what it needs. `atLeast`: the backend counts
 * the threshold itself as past it (department_confidence, machine.py); the rest need more. */
type Bar = { name: string; label: string; value: number; threshold?: number; risk?: boolean; atLeast?: boolean }

function pastThreshold(row: Bar): boolean {
  if (row.threshold === undefined) return false
  return row.atLeast ? row.value >= row.threshold : row.value > row.threshold
}

function bars(decision: Decision | null, screening: Screening | null, t: Thresholds | null): Bar[] {
  const source = decision ?? screening
  if (!source) return []
  const rows: Bar[] = [
    { name: 'safety_hazard', label: 'Safety risk', value: source.safety_hazard, threshold: t?.safety_hazard, risk: true },
    { name: 'wants_human', label: 'Wants an officer', value: source.wants_human, threshold: t?.wants_human, risk: true },
    { name: 'out_of_scope', label: 'Not a civic matter', value: source.out_of_scope, threshold: t?.out_of_scope, risk: true },
  ]
  if (decision) {
    rows.push(
      { name: 'department_confidence', label: `Department: ${departmentLabel(decision.department)}`, value: decision.department_confidence, threshold: t?.department_confidence, atLeast: true },
      { name: 'has_location', label: 'Gave a location', value: decision.has_location, threshold: t?.has_location },
      { name: 'repeat_complaint', label: 'Repeat complaint', value: decision.repeat_complaint, threshold: t?.repeat_complaint },
      { name: 'contains_personal_id', label: 'Said a personal number', value: decision.contains_personal_id },
    )
  }
  return rows
}

function DecisionBars({
  decision,
  screening,
  thresholds,
  joinedMidCall,
}: {
  decision: Decision | null
  screening: Screening | null
  thresholds: Thresholds | null
  joinedMidCall: boolean
}) {
  const rows = useMemo(() => bars(decision, screening, thresholds), [decision, screening, thresholds])
  if (rows.length === 0) {
    return (
      <p className="subtitle sv-bars-empty">
        {joinedMidCall
          ? 'The console joined during this call: its decision shows after the caller next speaks.'
          : 'No decision yet: waiting for the caller.'}
      </p>
    )
  }
  const urgency = (decision ?? screening)!.urgency
  const handOffMood = decision && thresholds?.hand_off_caller_states.includes(decision.caller_state)
  return (
    <section className="sv-bars" aria-label="Laya's decision">
      <h3>Laya's decision</h3>
      <p className="sv-bars-note">
        {screening && !decision ? 'Safety step; the rest is still running. ' : ''}
        Probability of yes. The upright mark is the threshold.
      </p>
      <p className="sv-chips">
        <span className={`sv-chip sv-urgency-${urgency}`}>Urgency: {urgency}</span>
        {decision && (
          <span className={`sv-chip${handOffMood ? ' sv-urgency-emergency' : ''}`}>Mood: {decision.caller_state}</span>
        )}
        {decision && <span className="sv-chip">{wardLabel(decision.ward)}</span>}
      </p>
      <ul>
        {rows.map((row) => {
          const over = pastThreshold(row)
          const threshold = row.threshold === undefined ? '' : `, threshold ${row.threshold.toFixed(2)}`
          return (
            <li key={row.name} className={`${row.risk ? 'is-risk' : ''} ${over ? (row.risk ? 'is-alarm' : 'is-met') : ''}`}>
              <span className="sv-bar-label">
                {row.label}
                {over && row.risk ? ' · past threshold' : ''}
              </span>
              <span className="sv-bar-value">{row.value.toFixed(2)}</span>
              <span
                className="sv-bar"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={row.value}
                aria-valuetext={`${row.value.toFixed(2)}${threshold}${over ? ', past it' : ''}`}
                aria-label={row.label}
              >
                <span className="sv-bar-fill" style={{ width: `${Math.round(row.value * 100)}%` }} />
                {row.threshold !== undefined && (
                  <span className="sv-bar-mark" style={{ left: `${row.threshold * 100}%` }} title={`Threshold ${row.threshold}`} />
                )}
              </span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function TicketTable({ tickets, error }: { tickets: StoredTicket[]; error: string | null }) {
  const [filter, setFilter] = useState<TicketFilter>(NO_FILTER)
  const shown = filterTickets(tickets, filter)
  const select = (key: keyof TicketFilter, options: [string, string][]) => (
    <label>
      {key[0].toUpperCase() + key.slice(1)}
      <select value={filter[key]} onChange={(e) => setFilter({ ...filter, [key]: e.target.value })}>
        <option value={ALL}>All</option>
        {options.map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
    </label>
  )
  return (
    <section className="sv-tickets" aria-label="Tickets">
      <div className="sv-tickets-head">
        <h2>Tickets{tickets.length > 0 ? ` (${tickets.length})` : ''}</h2>
        <div className="sv-filters">
          {select('department', Object.entries(DEPARTMENTS))}
          {select('ward', [...Object.keys(WARDS).map((id): [string, string] => [id, wardLabel(id)]), ['unknown', 'Unknown ward']])}
          {select('urgency', URGENCY_LEVELS.map((u): [string, string] => [u, u]))}
        </div>
      </div>
      {error && <p className="error">Tickets could not be loaded: {error}</p>}
      <div className="sv-table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Ticket</th>
              <th scope="col">Department</th>
              <th scope="col">Ward</th>
              <th scope="col">Urgency</th>
              <th scope="col">Where (caller's words)</th>
              <th scope="col">Filed</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((t) => (
              <tr key={t.number}>
                <td>{t.number}</td>
                <td>{departmentLabel(t.department)}</td>
                <td>{wardLabel(t.ward)}</td>
                <td>
                  <span className={`sv-chip sv-urgency-${t.urgency}`}>{t.urgency}</span>
                  {t.repeat_complaint && <span className="sv-chip">repeat</span>}
                </td>
                <td lang="kn">{t.location_text || '—'}</td>
                <td>{new Date(t.created_at).toLocaleString()}</td>
              </tr>
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={6} className="subtitle">
                  {tickets.length === 0 ? 'No tickets yet' : 'No tickets match these filters'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}
