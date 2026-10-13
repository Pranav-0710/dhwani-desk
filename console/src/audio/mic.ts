import { Downsampler, floatToInt16 } from './pcm'

/** Send to the backend in ~40 ms batches rather than one tiny frame per 128-sample block. */
const BATCH_SECONDS = 0.04

export interface Mic {
  stop: () => Promise<void>
}

/**
 * Captures the microphone, downsamples it to `targetRate` and hands over 16-bit PCM batches.
 * Echo cancellation and noise suppression are on, so the agent's own voice from the
 * speakers is mostly removed before it reaches speech recognition.
 */
export async function startMic(targetRate: number, onPcm: (pcm: Int16Array<ArrayBuffer>) => void): Promise<Mic> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  })
  const context = new AudioContext()
  await context.audioWorklet.addModule('/pcm-capture-worklet.js')
  const source = context.createMediaStreamSource(stream)
  const capture = new AudioWorkletNode(context, 'pcm-capture')
  const downsampler = new Downsampler(context.sampleRate, targetRate)
  const batchSize = Math.round(targetRate * BATCH_SECONDS)
  let pending: number[] = []

  capture.port.onmessage = (event: MessageEvent<Float32Array>) => {
    pending.push(...downsampler.process(event.data))
    if (pending.length >= batchSize) {
      onPcm(floatToInt16(Float32Array.from(pending)))
      pending = []
    }
  }
  source.connect(capture)

  return {
    stop: async () => {
      capture.port.onmessage = null
      source.disconnect()
      capture.disconnect()
      stream.getTracks().forEach((track) => track.stop())
      await context.close()
    },
  }
}
