import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Mic } from '../audio/mic'
import type { NoiseLevel } from '../audio/noise'
import { CallClient, type CallEvent, type CallSocket, type Player } from './call-client'

class FakeSocket implements CallSocket {
  binaryType: BinaryType = 'blob'
  readyState = 1
  onmessage: ((message: { data: unknown }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  sent: ArrayBuffer[] = []
  closedWith: number | undefined

  send(data: ArrayBuffer): void {
    this.sent.push(data)
  }

  close(code?: number): void {
    this.closedWith = code
    this.readyState = 3
    this.onclose?.()
  }

  serverSends(message: object): void {
    this.onmessage?.({ data: JSON.stringify(message) })
  }
}

class FakePlayer implements Player {
  queued = 0
  closed = false
  start(): void {}
  push(): void {}
  remaining(): number {
    return this.queued
  }
  async close(): Promise<void> {
    this.closed = true
  }
}

function setup() {
  const socket = new FakeSocket()
  const player = new FakePlayer()
  const events: CallEvent[] = []
  let clock = 0
  let micResolve: ((mic: Mic) => void) | null = null
  let onPcm: ((pcm: Int16Array<ArrayBuffer>) => void) | null = null
  const mic = {
    stopped: false,
    noise: [] as NoiseLevel[],
    setNoise: (level: NoiseLevel) => void mic.noise.push(level),
    stop: async () => void (mic.stopped = true),
  }

  const client = new CallClient((event) => events.push(event), {
    openSocket: () => socket,
    createPlayer: () => player,
    now: () => clock,
    // Mutes as the real mic does with no line noise: the caller's sound becomes silence.
    startMic: (_rate, callback, muted) => {
      onPcm = (pcm) => callback(muted() ? new Int16Array(pcm.length) : pcm)
      return new Promise<Mic>((resolve) => (micResolve = resolve))
    },
  })
  client.start('ws://test/ws/call')

  return {
    client,
    socket,
    player,
    events,
    mic,
    statuses: () => events.filter((e) => e.type === 'status').map((e) => e.status),
    grantMic: async () => {
      micResolve?.(mic)
      await flush()
    },
    speak: (samples: number[]) => onPcm?.(Int16Array.from(samples)),
    advance: (ms: number) => {
      clock += ms
      vi.advanceTimersByTime(ms)
    },
  }
}

/** Let pending promise callbacks run (setTimeout is faked in these tests). */
const flush = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve()
}

describe('CallClient', () => {
  beforeEach(() => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] }))
  afterEach(() => vi.useRealTimers())

  it('goes from connecting to listening once the mic is granted', async () => {
    const call = setup()
    call.socket.serverSends({ type: 'config', sample_rate: 8000, tts_sample_rate: 16000 })
    call.socket.serverSends({ type: 'connected' })
    await call.grantMic()

    expect(call.statuses()).toEqual(['connecting', 'listening'])
  })

  it('releases a mic granted after the caller already hung up', async () => {
    const call = setup()
    call.socket.serverSends({ type: 'connected' })
    await call.client.hangUp() // while the permission prompt is still open
    await call.grantMic()

    expect(call.mic.stopped).toBe(true)
    expect(call.statuses()).toEqual(['connecting', 'ended'])
    expect(call.player.closed).toBe(true)
  })

  it('sends silence, not sound, while the agent speaks and until its audio has played', async () => {
    const call = setup()
    call.socket.serverSends({ type: 'connected' })
    await call.grantMic()

    call.socket.serverSends({ type: 'agent_audio_start', sample_rate: 16000 })
    call.speak([100, 200])
    call.player.queued = 1 // one second still in the speakers when the server finishes
    call.socket.serverSends({ type: 'agent_audio_end' })
    call.speak([300, 400])

    call.advance(1200) // 1 s of playback + 300 ms tail not yet over
    call.speak([500, 600])
    expect(call.statuses().at(-1)).toBe('agent-speaking')

    call.player.queued = 0
    call.advance(200)
    call.speak([700, 800])

    const sent = call.socket.sent.map((buffer) => Array.from(new Int16Array(buffer)))
    expect(sent).toEqual([
      [0, 0],
      [0, 0],
      [0, 0],
      [700, 800],
    ])
    expect(call.statuses()).toEqual(['connecting', 'listening', 'agent-speaking', 'listening'])
  })

  it('reports a denied mic and ends the call without claiming to listen', async () => {
    const socket = new FakeSocket()
    const events: CallEvent[] = []
    const client = new CallClient((event) => events.push(event), {
      openSocket: () => socket,
      createPlayer: () => new FakePlayer(),
      startMic: async () => {
        throw new Error('NotAllowedError')
      },
    })
    client.start('ws://test/ws/call')
    socket.serverSends({ type: 'connected' })
    await flush()

    const statuses = events.filter((e) => e.type === 'status').map((e) => e.status)
    expect(statuses).toEqual(['connecting', 'ended'])
    expect(events.some((e) => e.type === 'error' && e.message.includes('NotAllowedError'))).toBe(
      true,
    )
  })

  it('puts line noise on the mic when it opens and whenever it changes', async () => {
    const call = setup()
    call.client.setNoise('heavy') // chosen before the permission prompt is answered
    call.socket.serverSends({ type: 'connected' })
    await call.grantMic()
    call.client.setNoise('off')

    expect(call.mic.noise).toEqual(['heavy', 'off'])
  })

  it('does not print empty brackets when Prisma closes without a reason', () => {
    const call = setup()
    call.socket.serverSends({ type: 'closed', reason: '' })

    expect(call.events.at(-1)).toEqual({ type: 'error', message: 'Speech service ended the call' })
  })
})
