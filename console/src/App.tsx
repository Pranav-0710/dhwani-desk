import { useEffect, useRef, useState } from 'react'
import { NOISE_LEVELS, type NoiseLevel } from './audio/noise'
import { CallClient, type CallEvent, type CallStatus } from './call/call-client'
import { callErrorText } from './call/error-text'

const NOISE_LABEL: Record<NoiseLevel, string> = {
  off: 'Off',
  light: 'Quiet street',
  heavy: 'Busy street',
}

type Line = { id: number; who: 'caller' | 'agent'; text: string; latencyMs?: number | null }

/** Kannada first, for the caller; English under it. */
const STATUS_LABEL: Record<CallStatus, { kn: string; en: string }> = {
  idle: { kn: 'ಕರೆ ಮಾಡಲು ಸಿದ್ಧ', en: 'Ready to call' },
  connecting: { kn: 'ಸಂಪರ್ಕಿಸಲಾಗುತ್ತಿದೆ…', en: 'Connecting…' },
  listening: { kn: 'ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡಿ', en: 'Listening — speak in Kannada' },
  'agent-speaking': { kn: 'ಸಹಾಯವಾಣಿ ಮಾತನಾಡುತ್ತಿದೆ', en: 'Agent is speaking' },
  ended: { kn: 'ಕರೆ ಮುಗಿದಿದೆ', en: 'Call ended' },
}

export default function App() {
  const [status, setStatus] = useState<CallStatus>('idle')
  const [lines, setLines] = useState<Line[]>([])
  const [error, setError] = useState<string | null>(null)
  const [noise, setNoise] = useState<NoiseLevel>('off')
  const client = useRef<CallClient | null>(null)
  const nextId = useRef(0)
  const transcript = useRef<HTMLOListElement>(null)

  const inCall = status === 'connecting' || status === 'listening' || status === 'agent-speaking'

  useEffect(() => () => void client.current?.hangUp(), [])

  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight })
  }, [lines.length])

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
    client.current.setNoise(noise)
    client.current.start()
  }

  async function hangUp() {
    await client.current?.hangUp()
    client.current = null
  }

  function chooseNoise(level: NoiseLevel) {
    setNoise(level)
    client.current?.setNoise(level)
  }

  return (
    <main className="phone">
      <header>
        <h1 lang="kn">ನಗರ ಸೇವಾ ಸಹಾಯವಾಣಿ</h1>
        <p className="subtitle">
          Nagara Seva helpline
          <br />
          City Corporation of Hosapura · demo line
        </p>
      </header>

      <div className={`status status-${status}`} role="status">
        <span className="status-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span>
          <span className="status-main" lang="kn">
            {STATUS_LABEL[status].kn}
          </span>
          <span className="status-sub">{STATUS_LABEL[status].en}</span>
        </span>
      </div>

      {error && (
        <p className="error" role="alert">
          {callErrorText(error)}
        </p>
      )}

      {lines.length === 0 && (
        <p className="hint">
          {!inCall && !error && (
            <>
              <span className="hint-main" lang="kn">
                ಕರೆ ಮಾಡಿ, ಕನ್ನಡದಲ್ಲಿ ನಿಮ್ಮ ದೂರು ಹೇಳಿ.
              </span>
              Press Call, then say your complaint in Kannada.
            </>
          )}
        </p>
      )}

      <ol className="transcript" ref={transcript} aria-label="Conversation">
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

      <fieldset className="noise">
        <legend>Line noise</legend>
        {NOISE_LEVELS.map((level) => (
          <label key={level}>
            <input
              type="radio"
              name="noise"
              value={level}
              checked={noise === level}
              onChange={() => chooseNoise(level)}
            />
            {NOISE_LABEL[level]}
          </label>
        ))}
      </fieldset>

      <button
        type="button"
        className={inCall ? 'hang-up' : 'call'}
        onClick={inCall ? hangUp : startCall}
      >
        <span lang="kn">{inCall ? 'ಕರೆ ಮುಗಿಸಿ' : status === 'ended' ? 'ಮತ್ತೆ ಕರೆ ಮಾಡಿ' : 'ಕರೆ ಮಾಡಿ'}</span>
        {' · '}
        {inCall ? 'Hang up' : status === 'ended' ? 'Call again' : 'Call'}
      </button>
    </main>
  )
}
