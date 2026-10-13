import { int16BytesToFloat } from './pcm'

/** A little head start so the first chunk isn't clipped while the scheduler wakes up. */
const START_DELAY_S = 0.05

/**
 * Plays the agent's streamed 16-bit PCM gap-free: each chunk is scheduled to start exactly
 * where the previous one ends.
 */
export class PcmPlayer {
  private readonly context = new AudioContext()
  private sampleRate = 16000
  private nextStart = 0
  private sources = new Set<AudioBufferSourceNode>()

  start(sampleRate: number): void {
    this.sampleRate = sampleRate
    if (this.context.state === 'suspended') {
      void this.context.resume()
    }
  }

  push(pcm: ArrayBuffer): void {
    const samples = int16BytesToFloat(pcm)
    if (samples.length === 0) return
    const buffer = this.context.createBuffer(1, samples.length, this.sampleRate)
    buffer.copyToChannel(samples, 0)
    const source = this.context.createBufferSource()
    source.buffer = buffer
    source.connect(this.context.destination)
    const startAt = Math.max(this.nextStart, this.context.currentTime + START_DELAY_S)
    source.start(startAt)
    this.nextStart = startAt + buffer.duration
    this.sources.add(source)
    source.onended = () => this.sources.delete(source)
  }

  /** Seconds of agent audio still to come out of the speakers, including output latency
   * (Bluetooth headsets can add 150–300 ms). */
  remaining(): number {
    const queued = this.nextStart - this.context.currentTime
    if (queued <= 0) return 0
    return queued + (this.context.outputLatency || this.context.baseLatency || 0)
  }

  /** Cut the agent off (used for barge-in in #10). */
  stop(): void {
    this.sources.forEach((source) => source.stop())
    this.sources.clear()
    this.nextStart = 0
  }

  async close(): Promise<void> {
    this.stop()
    await this.context.close()
  }
}
