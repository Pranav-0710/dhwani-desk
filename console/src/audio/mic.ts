import { NoiseMixer, type NoiseLevel } from './noise'
import { Downsampler, floatToInt16 } from './pcm'

/** Send to the backend in ~40 ms batches rather than one tiny frame per 128-sample block. */
const BATCH_SECONDS = 0.04

export interface Mic {
  /** Background line noise mixed into what is sent from now on (`noise.ts`). */
  setNoise: (level: NoiseLevel) => void
  stop: () => Promise<void>
}

export type StartMic = (
  targetRate: number,
  onPcm: (pcm: Int16Array<ArrayBuffer>) => void,
) => Promise<Mic>

/**
 * Captures the microphone, downsamples it to `targetRate`, mixes in any background line noise
 * and hands over 16-bit PCM batches.
 * Echo cancellation and noise suppression are on, so the agent's own voice from the
 * speakers is mostly removed before it reaches speech recognition. If any setup step
 * fails, everything already acquired is released before the error is rethrown.
 */
export const startMic: StartMic = async (targetRate, onPcm) => {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  })
  const context = new AudioContext()
  const release = async () => {
    stream.getTracks().forEach((track) => track.stop())
    if (context.state !== 'closed') await context.close()
  }

  try {
    // Created after the permission prompt, so outside the click: Safari starts it suspended.
    if (context.state === 'suspended') await context.resume()
    await context.audioWorklet.addModule(`${import.meta.env.BASE_URL}pcm-capture-worklet.js`)
    const source = context.createMediaStreamSource(stream)
    const capture = new AudioWorkletNode(context, 'pcm-capture', { numberOfOutputs: 0 })
    const downsampler = new Downsampler(context.sampleRate, targetRate)
    const noise = new NoiseMixer()
    const batchSize = Math.round(targetRate * BATCH_SECONDS)
    let pending: number[] = []

    capture.port.onmessage = (event: MessageEvent<Float32Array>) => {
      pending.push(...downsampler.process(event.data))
      if (pending.length >= batchSize) {
        onPcm(floatToInt16(noise.mix(Float32Array.from(pending))))
        pending = []
      }
    }
    source.connect(capture)

    return {
      setNoise: (level) => {
        noise.level = level
      },
      stop: async () => {
        capture.port.onmessage = null
        source.disconnect()
        await release()
      },
    }
  } catch (error) {
    await release()
    throw error
  }
}
