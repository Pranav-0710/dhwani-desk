import { useEffect, useRef, useState } from 'react'
import { CallClient, type CallEvent, type CallStatus } from './call/call-client'

type Line = { id: number; who: 'caller' | 'agent'; text: string; latencyMs?: number | null }

const STATUS_LABEL: Record<CallStatus, string> = {
  idle: 'Ready to call',
  connecting: 'Connecting…',
  listening: 'Listening — speak in Kannada',
  'agent-speaking': 'Agent is speaking',
  ended: 'Call ended',
}

export default function App() {
  const [status, setStatus] = useState<CallStatus>('idle')
  const [lines, setLines] = useState<Line[]>([])
  const [error, setError] = useState<string | null>(null)
  const client = useRef<CallClient | null>(null)
  const nextId = useRef(0)

  const inCall = status === 'connecting' || status === 'listening' || status === 'agent-speaking'

  useEffect(() => () => void client.current?.hangUp(), [])

  function onEvent(event: CallEvent) {
    switch (event.type) {
      case 'status':
        setStatus(event.status)
        break
      case 'caller':
      case 'agent': {
        const line: Line = {
          id: nextId.current++,
          who: event.type,
          text: event.text,
          latencyMs: event.type === 'caller' ? event.latencyMs : undefined,
        }
        setLines((previous) => [...previous, line])
        // A new agent line means any earlier voice hiccup is over.
        if (event.type === 'agent') setError(null)
        break
      }
      case 'error':
        setError(event.message)
        break
    }
  }

  function startCall() {
    setLines([])
    setError(null)
    client.current = new CallClient(onEvent)
    client.current.start()
  }

  async function hangUp() {
    await client.current?.hangUp()
    client.current = null
  }

  return (
    <main className="phone">
      <header>
        <h1>Nagara Seva helpline</h1>
        <p className="subtitle">City Corporation of Hosapura · demo line</p>
      </header>

      <section className={`status status-${status}`} aria-live="polite">
        {STATUS_LABEL[status]}
      </section>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <ol className="transcript" aria-label="Conversation">
        {lines.map((line) => (
          <li key={line.id} className={line.who}>
            <span className="who">{line.who === 'caller' ? 'You' : 'Agent'}</span>
            <span className="text" lang="kn">
              {line.text}
            </span>
            {line.latencyMs != null && <span className="latency">{line.latencyMs} ms</span>}
          </li>
        ))}
      </ol>

      <button
        type="button"
        className={inCall ? 'hang-up' : 'call'}
        onClick={inCall ? hangUp : startCall}
      >
        {inCall ? 'Hang up' : 'Call'}
      </button>
    </main>
  )
}
