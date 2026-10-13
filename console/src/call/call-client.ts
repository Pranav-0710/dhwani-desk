import { startMic, type Mic } from '../audio/mic'
import { PcmPlayer } from '../audio/player'

/** Keep the mic muted this long after the agent finishes, so its tail isn't re-heard. */
const ECHO_TAIL_S = 0.3

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

/**
 * One phone call over `/ws/call`: mic audio up, transcripts and agent speech down.
 *
 * Half duplex for now: while the agent is speaking (plus a short tail) the mic is not sent,
 * so the echo bot never answers its own voice. Barge-in (#10) replaces this.
 */
export class CallClient {
  private socket: WebSocket | null = null
  private mic: Mic | null = null
  private player: PcmPlayer | null = null
  private agentSpeaking = false
  private mutedUntil = 0
  private micRate = 8000
  private ended = false
  private readonly onEvent: (event: CallEvent) => void

  constructor(onEvent: (event: CallEvent) => void) {
    this.onEvent = onEvent
  }

  start(url = callUrl()): void {
    this.onEvent({ type: 'status', status: 'connecting' })
    this.player = new PcmPlayer()
    const socket = new WebSocket(url)
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
        await this.openMic()
        this.onEvent({ type: 'status', status: 'listening' })
        break
      case 'transcript':
        this.onEvent({ type: 'caller', text: message.text, latencyMs: message.latency_ms })
        break
      case 'agent_text':
        this.onEvent({ type: 'agent', text: message.text })
        break
      case 'agent_audio_start':
        this.agentSpeaking = true
        this.player?.start(message.sample_rate)
        this.onEvent({ type: 'status', status: 'agent-speaking' })
        break
      case 'agent_audio_end':
        this.agentSpeaking = false
        this.onEvent({ type: 'status', status: 'listening' })
        break
      case 'tts_error':
        this.onEvent({ type: 'error', message: `Agent voice failed: ${message.message}` })
        break
      case 'closed':
        this.onEvent({ type: 'error', message: `Speech service ended the call (${message.reason})` })
        break
      case 'error':
        this.onEvent({ type: 'error', message: message.message })
        break
    }
  }

  private async openMic(): Promise<void> {
    try {
      this.mic = await startMic(this.micRate, (pcm) => {
        const remaining = this.player?.remaining() ?? 0
        if (remaining > 0) {
          this.mutedUntil = performance.now() + (remaining + ECHO_TAIL_S) * 1000
        }
        const muted = this.agentSpeaking || performance.now() < this.mutedUntil
        if (!muted && this.socket?.readyState === WebSocket.OPEN) {
          this.socket.send(pcm.buffer)
        }
      })
    } catch (error) {
      this.onEvent({ type: 'error', message: `Microphone unavailable: ${String(error)}` })
      this.socket?.close(1000)
    }
  }

  private async finish(): Promise<void> {
    if (this.ended) return
    this.ended = true
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
