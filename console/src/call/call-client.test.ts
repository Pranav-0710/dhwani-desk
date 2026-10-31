import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Mic } from '../audio/mic'
import type { NoiseLevel } from '../audio/noise'
import {
  BARGE_IN_MS,
  BARGE_IN_RMS,
  CallClient,
  type CallEvent,
  type CallSocket,
  type Player,
} from './call-client'

class FakeSocket implements CallSocket {
  binaryType: BinaryType = 'blob'
  readyState = 1
  onmessage: ((message: { data: unknown }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  sent: ArrayBuffer[] = []
  texts: string[] = []
  closedWith: number | undefined

  send(data: ArrayBuffer | string): void {
    if (typeof data === 'string') this.texts.push(data)
    else this.sent.push(data)
  }

  serverSendsAudio(): void {
    this.onmessage?.({ data: new ArrayBuffer(4) })
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
  pushed = 0
  stops = 0
  closed = false
  start(): void {}
  push(): void {
    this.pushed += 1
  }
  remaining(): number {
    return this.queued
  }
  stop(): void {
    this.stops += 1
    this.queued = 0
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
  let onLevel: ((rms: number) => void) | undefined
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
    startMic: (_rate, callback, muted, level) => {
      onPcm = (pcm) => callback(muted() ? new Int16Array(pcm.length) : pcm)
      onLevel = level
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
    /** The caller's level for one 40 ms mic batch, then the clock moves on. */
    level: (rms: number) => {
      onLevel?.(rms)
      clock += 40
    },
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
  describe('barge-in', () => {
    async function agentSpeaking() {
      const call = setup()
      call.socket.serverSends({ type: 'connected' })
      await call.grantMic()
      call.socket.serverSends({ type: 'agent_audio_start', sample_rate: 16000 })
      call.player.queued = 5
      return call
    }

    it('stops playback, opens the mic and tells the server when the caller talks over the agent', async () => {
      const call = await agentSpeaking()
      const batches = Math.ceil(BARGE_IN_MS / 40) + 1

      for (let i = 0; i < batches; i++) call.level(BARGE_IN_RMS * 2)

      expect(call.player.stops).toBe(1)
      expect(call.socket.texts).toEqual(['barge_in'])
      expect(call.statuses().at(-1)).toBe('listening')
      // Within the 300 ms the issue asks for.
      expect(batches * 40).toBeLessThanOrEqual(300)
      // The caller's sound now goes to the server.
      call.speak([5, 6])
      expect(Array.from(new Int16Array(call.socket.sent.at(-1)!))).toEqual([5, 6])
    })

    it("drops the cut line's late audio and does not mute the caller for its tail", async () => {
      const call = await agentSpeaking()
      for (let i = 0; i < 6; i++) call.level(1)

      call.socket.serverSendsAudio()
      call.socket.serverSends({ type: 'agent_audio_end', interrupted: true })
      call.speak([7])

      expect(call.player.pushed).toBe(0)
      expect(Array.from(new Int16Array(call.socket.sent.at(-1)!))).toEqual([7])
      // The next line plays as usual.
      call.socket.serverSends({ type: 'agent_audio_start', sample_rate: 16000 })
      call.socket.serverSendsAudio()
      expect(call.player.pushed).toBe(1)
    })

    it('ignores a short or quiet sound, and any sound while the agent is silent', async () => {
      const call = await agentSpeaking()
      call.level(1) // a click
      call.level(0)
      for (let i = 0; i < 10; i++) call.level(BARGE_IN_RMS / 2) // leftover echo
      call.player.queued = 0 // played out
      call.socket.serverSends({ type: 'agent_audio_end' })
      call.advance(1000)
      for (let i = 0; i < 10; i++) call.level(1) // the caller's own turn

      expect(call.player.stops).toBe(0)
      expect(call.socket.texts).toEqual([])
    })
  })
})
