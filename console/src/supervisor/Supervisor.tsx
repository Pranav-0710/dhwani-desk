import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { ConsoleClient, type Command, type ConsoleStatus } from './console-client'
import { DEPARTMENTS, URGENCY_LEVELS, WARDS, departmentLabel, wardLabel } from './labels'
import {
  EMPTY,
  REASON_LABEL,
  applyMessage,
  needsOfficer,
  type ConsoleState,
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

function saveToken(token: string): void {
  try {
    sessionStorage.setItem(TOKEN_KEY, token)
  } catch {
    // Private windows can refuse storage; the officer just types the token again.
  }
}

export default function Supervisor() {
  const [token, setToken] = useState(readToken)
  const [state, dispatch] = useReducer(
    (current: ConsoleState, message: ServerMessage) => applyMessage(current, message),
    EMPTY,
  )
  const [status, setStatus] = useState<ConsoleStatus>('closed')
  const [problem, setProblem] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [tickets, setTickets] = useState<StoredTicket[]>([])
  const client = useRef<ConsoleClient | null>(null)

  const refreshTickets = useCallback(() => {
    if (!token) return
    fetchTickets(token)
      .then(setTickets)
      .catch((error: Error) => setProblem(error.message))
  }, [token])

  useEffect(() => {
    if (!token) return
    const connection = new ConsoleClient(
      (message) => {
        dispatch(message)
        if (message.type === 'call_closed') refreshTickets()
      },
      (next, detail) => {
        setStatus(next)
        if (next === 'refused') setProblem(detail ?? 'The console token was refused')
        if (next === 'connected') setProblem(null)
      },
    )
    connection.connect(token)
    client.current = connection
    return () => {
      connection.close()
      client.current = null
    }
  }, [token, refreshTickets])

  useEffect(() => {
    if (status !== 'connected') return
    refreshTickets()
    const timer = setInterval(refreshTickets, TICKETS_REFRESH_MS)
    return () => clearInterval(timer)
  }, [status, refreshTickets])

  // Follow the call that most needs an officer, unless one is already chosen.
  const calls = state.order.map((id) => state.calls[id])
  const waiting = calls.filter(needsOfficer)
  const current = (selected && state.calls[selected]) || waiting[0] || calls[0] || null

  if (!token || status === 'refused') {
    return (
      <TokenGate
        problem={status === 'refused' ? problem : null}
        onSubmit={(value) => {
          saveToken(value)
          setStatus('connecting')
          setToken(value)
        }}
      />
    )
  }

  return (
    <div className="supervisor">
      <header className="sv-header">
        <div>
          <h1>Nagara Seva supervisor console</h1>
          <p className="subtitle">City Corporation of Hosapura · live calls and tickets</p>
        </div>
        <span className={`sv-status sv-status-${status}`}>{STATUS_LABEL[status]}</span>
      </header>

      {waiting.length > 0 && (
        <button className="sv-alert" onClick={() => setSelected(waiting[0].id)}>
          {waiting.length === 1 ? 'A call needs' : `${waiting.length} calls need`} an officer:{' '}
          {REASON_LABEL[waiting[0].reason ?? ''] ?? waiting[0].reason}
        </button>
      )}
      {problem && <p className="error">{problem}</p>}

      <div className="sv-main">
        <CallList calls={calls} current={current?.id ?? null} onSelect={setSelected} />
        {current ? (
          <CallView
            key={current.id}
            call={current}
            thresholds={state.thresholds}
            send={(command, text) =>
              client.current?.send(command, current.id, text) ??
              Promise.reject(new Error('The console is not connected'))
            }
          />
        ) : (
          <section className="sv-call sv-empty">No live calls. They appear here as they start.</section>
        )}
      </div>

      <TicketTable tickets={tickets} />
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
      <p className="subtitle">Enter the console token (CONSOLE_TOKEN in the backend's .env).</p>
      <label>
        Console token
        <input type="password" autoComplete="off" value={value} onChange={(e) => setValue(e.target.value)} />
      </label>
      {problem && <p className="error">{problem}</p>}
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
      <h2>Calls</h2>
      {calls.length === 0 && <p className="subtitle">None yet</p>}
      <ul>
        {calls.map((call) => (
          <li key={call.id}>
            <button
              className={[
                'sv-call-item',
                call.id === current ? 'is-current' : '',
                needsOfficer(call) ? 'is-waiting' : '',
                call.ended ? 'is-ended' : '',
              ].join(' ')}
              aria-current={call.id === current}
              onClick={() => onSelect(call.id)}
            >
              <span className="sv-call-time">{new Date(call.startedAt).toLocaleTimeString()}</span>
              <span>{departmentLabel(call.slots.department)}</span>
              <span className="sv-call-state">{stateLabel(call)}</span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  )
}

function CallView({
  call,
  thresholds,
  send,
}: {
  call: LiveCall
  thresholds: Thresholds | null
  send: (command: Command, text?: string) => Promise<void>
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const transcript = useRef<HTMLOListElement>(null)

  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight })
  }, [call.lines.length])

  async function run(command: Command, text?: string) {
    setBusy(true)
    setError(null)
    try {
      await send(command, text)
      if (command === 'say') setDraft('')
    } catch (failure) {
      setError((failure as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const live = !call.ended && call.state !== 'closed'
  return (
    <section className="sv-call" aria-label="Selected call">
      <div className="sv-call-head">
        <div>
          <h2>{departmentLabel(call.slots.department)}</h2>
          <p className="subtitle">
            {wardLabel(call.slots.ward)} · {stateLabel(call)}
            {call.reason ? ` · ${REASON_LABEL[call.reason] ?? call.reason}` : ''}
          </p>
        </div>
        {call.ticket && <span className="sv-ticket-chip">Ticket #{call.ticket.number}</span>}
      </div>

      {needsOfficer(call) && (
        <p className="sv-call-alert" role="alert">
          {REASON_LABEL[call.reason ?? ''] ?? 'Handed off'}: take the call.
        </p>
      )}

      <ol className="sv-transcript" ref={transcript} aria-label="Transcript" aria-live="polite">
        {call.lines.map((line, index) => (
          <li key={index} className={`sv-line sv-${line.who}`}>
            <span className="who">{line.who === 'officer' ? 'Officer (agent voice)' : line.who === 'caller' ? 'Caller' : 'Agent'}</span>
            <span className="text">{line.text}</span>
          </li>
        ))}
      </ol>

      <DecisionBars decision={call.decision} screening={call.screening} thresholds={thresholds} />

      <div className="sv-controls">
        {!call.officer ? (
          <button className="call" disabled={busy || !live} onClick={() => run('take_over')}>
            Take over
          </button>
        ) : (
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
              value={draft}
              maxLength={MAX_OFFICER_LINE}
              placeholder="Type in Kannada; the agent's voice says it"
              onChange={(e) => setDraft(e.target.value)}
              disabled={busy || !live}
            />
            <button className="call" type="submit" disabled={busy || !live || !draft.trim()}>
              Say
            </button>
          </form>
        )}
        <button className="sv-secondary" disabled={busy || !live || call.state !== 'handed_off'} onClick={() => run('hand_back')}>
          Hand back to the agent
        </button>
      </div>
      {error && <p className="error">{error}</p>}
    </section>
  )
}

/** `risk`: past its threshold the call is handed off or redirected, so it is shown as an alarm.
 * The others past theirs only mean the agent has what it needs. */
type Bar = { name: string; label: string; value: number; threshold?: number; risk?: boolean }

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
      { name: 'department_confidence', label: `Department: ${departmentLabel(decision.department)}`, value: decision.department_confidence, threshold: t?.department_confidence },
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
}: {
  decision: Decision | null
  screening: Screening | null
  thresholds: Thresholds | null
}) {
  const rows = useMemo(() => bars(decision, screening, thresholds), [decision, screening, thresholds])
  if (rows.length === 0) return <p className="subtitle sv-bars-empty">No decision yet: waiting for the caller.</p>
  const urgency = (decision ?? screening)!.urgency
  return (
    <section className="sv-bars" aria-label="Laya's decision">
      <h3>
        Decision {screening && !decision ? '(safety step; the rest is still running)' : ''}
      </h3>
      <p className="sv-chips">
        <span className={`sv-chip sv-urgency-${urgency}`}>Urgency: {urgency}</span>
        {decision && <span className="sv-chip">Mood: {decision.caller_state}</span>}
        {decision && <span className="sv-chip">{wardLabel(decision.ward)}</span>}
      </p>
      <ul>
        {rows.map((row) => {
          const over = row.threshold !== undefined && row.value > row.threshold
          return (
            <li key={row.name} className={over ? (row.risk ? 'is-alarm' : 'is-met') : ''}>
              <span className="sv-bar-label">{row.label}</span>
              <span
                className="sv-bar"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={row.value}
                aria-label={row.label}
              >
                <span className="sv-bar-fill" style={{ width: `${Math.round(row.value * 100)}%` }} />
                {row.threshold !== undefined && (
                  <span className="sv-bar-mark" style={{ left: `${row.threshold * 100}%` }} title={`Threshold ${row.threshold}`} />
                )}
              </span>
              <span className="sv-bar-value">{row.value.toFixed(2)}</span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function TicketTable({ tickets }: { tickets: StoredTicket[] }) {
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
        <h2>Tickets</h2>
        <div className="sv-filters">
          {select('department', Object.entries(DEPARTMENTS))}
          {select('ward', [...Object.keys(WARDS).map((id): [string, string] => [id, wardLabel(id)]), ['unknown', 'Unknown ward']])}
          {select('urgency', URGENCY_LEVELS.map((u): [string, string] => [u, u]))}
        </div>
      </div>
      <div className="sv-table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">#</th>
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
