import { startMic, type Mic, type StartMic } from '../audio/mic'
import { PcmPlayer } from '../audio/player'

/** Keep the mic muted this long after the agent's audio ends, so its tail isn't re-heard. */
const ECHO_TAIL_S = 0.3
/** WebSocket.OPEN, spelled out so the call logic runs without a browser. */
const SOCKET_OPEN = 1

export type CallStatus = 'idle' | 'connecting' | 'listening' | 'agent-speaking' | 'ended'

export type CallEvent =
  | { type: 'status'; status: CallStatus }
  | { type: 'caller'; text: string; latencyMs: number | null }
  | { type: 'agent'; text: string }
  | { type: 'error'; message: string }

type ServerMessage =
  | { type: 'config'; sample_rate: number; tts_sample_rate: number }
  | { type: 'connected' }
  | { type: 'processing' }
  | { type: 'transcript'; text: string; latency_ms: number | null }
  | { type: 'agent_text'; text: string }
  | { type: 'agent_audio_start'; sample_rate: number }
  | { type: 'agent_audio_end' }
  | { type: 'tts_error'; message: string }
  | { type: 'closed'; reason: string }
  | { type: 'error'; message: string }

/** What the client needs from the audio player; PcmPlayer in the browser. */
export interface Player {
  start(sampleRate: number): void
  push(pcm: ArrayBuffer): void
  remaining(): number
  close(): Promise<void>
}

/** What the client needs from a WebSocket; injectable so the call logic can be tested. */
export interface CallSocket {
  binaryType: BinaryType
  readyState: number
  onmessage: ((message: { data: unknown }) => void) | null
  onclose: (() => void) | null
  onerror: (() => void) | null
  send(data: ArrayBuffer): void
  close(code?: number): void
}

export interface CallDeps {
  openSocket: (url: string) => CallSocket
  startMic: StartMic
  createPlayer: () => Player
  now: () => number
}

const browserDeps: CallDeps = {
  openSocket: (url) => new WebSocket(url) as unknown as CallSocket,
  startMic,
  createPlayer: () => new PcmPlayer(),
  now: () => performance.now(),
}

/**
 * One phone call over `/ws/call`: mic audio up, transcripts and agent speech down.
 *
 * Half duplex for now: while the agent is speaking (until its audio has actually played,
 * plus a short tail) the mic sends silence instead of sound, so the echo bot never answers
 * its own voice while Prisma still gets a continuous real-time stream. Barge-in (#10)
 * replaces this. A CallClient is single-use: create a new one per call.
 */
export class CallClient {
  private socket: CallSocket | null = null
  private mic: Mic | null = null
  private player: Player | null = null
  private agentSpeaking = false
  private mutedUntil = 0
  private listeningTimer: ReturnType<typeof setTimeout> | null = null
  private micRate = 8000
  private ended = false
  private readonly onEvent: (event: CallEvent) => void
  private readonly deps: CallDeps

  constructor(onEvent: (event: CallEvent) => void, deps: Partial<CallDeps> = {}) {
    this.onEvent = onEvent
    this.deps = { ...browserDeps, ...deps }
  }

  start(url = callUrl()): void {
    this.emitStatus('connecting')
    // Created inside the click handler, so the browser lets it play sound.
    this.player = this.deps.createPlayer()
    const socket = this.deps.openSocket(url)
    socket.binaryType = 'arraybuffer'
    socket.onmessage = (message) => {
      if (typeof message.data === 'string') {
        void this.handle(JSON.parse(message.data) as ServerMessage)
      } else {
        this.player?.push(message.data as ArrayBuffer)
      }
    }
    socket.onclose = () => void this.finish()
    socket.onerror = () => this.onEvent({ type: 'error', message: 'Lost connection to the backend' })
    this.socket = socket
  }

  async hangUp(): Promise<void> {
    this.socket?.close(1000)
    await this.finish()
  }

  private async handle(message: ServerMessage): Promise<void> {
    switch (message.type) {
      case 'config':
        this.micRate = message.sample_rate
        break
      case 'connected':
        if (await this.openMic()) this.emitStatus('listening')
        break
      case 'transcript':
        this.onEvent({ type: 'caller', text: message.text, latencyMs: message.latency_ms })
        break
      case 'agent_text':
        this.onEvent({ type: 'agent', text: message.text })
        break
      case 'agent_audio_start':
        this.agentSpeaking = true
        this.clearListeningTimer()
        this.player?.start(message.sample_rate)
        this.emitStatus('agent-speaking')
        break
      case 'agent_audio_end':
        this.onAgentAudioEnd()
        break
      case 'tts_error':
        this.onEvent({ type: 'error', message: `Agent voice failed: ${message.message}` })
        break
      case 'closed':
        this.onEvent({
          type: 'error',
          message: `Speech service ended the call${message.reason ? ` (${message.reason})` : ''}`,
        })
        break
      case 'error':
        this.onEvent({ type: 'error', message: message.message })
        break
    }
  }

  /** The server has sent the last chunk; the speakers may still be playing it. */
  private onAgentAudioEnd(): void {
    this.agentSpeaking = false
    const quietInMs = ((this.player?.remaining() ?? 0) + ECHO_TAIL_S) * 1000
    this.mutedUntil = this.deps.now() + quietInMs
    this.clearListeningTimer()
    this.listeningTimer = setTimeout(() => {
      this.listeningTimer = null
      if (!this.agentSpeaking) this.emitStatus('listening')
    }, quietInMs)
  }

  private async openMic(): Promise<boolean> {
    let mic: Mic
    try {
      mic = await this.deps.startMic(this.micRate, (pcm) => this.sendMic(pcm))
    } catch (error) {
      this.onEvent({ type: 'error', message: `Microphone unavailable: ${String(error)}` })
      this.socket?.close(1000)
      return false
    }
    // The permission prompt can take a while; the call may have ended meanwhile.
    if (this.ended) {
      await mic.stop()
      return false
    }
    this.mic = mic
    return true
  }

  private sendMic(pcm: Int16Array<ArrayBuffer>): void {
    if (this.socket?.readyState !== SOCKET_OPEN) return
    const muted = this.agentSpeaking || this.deps.now() < this.mutedUntil
    this.socket.send(muted ? new ArrayBuffer(pcm.byteLength) : pcm.buffer)
  }

  private emitStatus(status: CallStatus): void {
    if (this.ended && status !== 'ended') return
    this.onEvent({ type: 'status', status })
  }

  private clearListeningTimer(): void {
    if (this.listeningTimer !== null) clearTimeout(this.listeningTimer)
    this.listeningTimer = null
  }

  private async finish(): Promise<void> {
    if (this.ended) return
    this.ended = true
    this.clearListeningTimer()
    const mic = this.mic
    const player = this.player
    this.mic = null
    this.player = null
    this.socket = null
    await mic?.stop()
    await player?.close()
    this.onEvent({ type: 'status', status: 'ended' })
  }
}

function callUrl(): string {
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${window.location.host}/ws/call`
}
