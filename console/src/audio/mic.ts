import { NoiseMixer, type NoiseLevel } from './noise'
import { Downsampler, floatToInt16 } from './pcm'

/** Send to the backend in ~40 ms batches rather than one tiny frame per 128-sample block. */
const BATCH_SECONDS = 0.04
/** How much of the caller's own sound is kept while muted, for a barge-in to send. Barge-in
 * needs 160 ms of loud speech before it opens the mic, so without this the start is lost. */
const RECENT_BATCHES = Math.round(0.3 / BATCH_SECONDS)

export interface Mic {
  /** Background line noise mixed into what is sent from now on (`noise.ts`). */
  setNoise: (level: NoiseLevel) => void
  /** The caller's own sound from the last ~300 ms of muted batches, oldest first; emptied
   * by the call. Sent on a barge-in, so the first syllables are not lost. */
  takeRecent?: () => Int16Array<ArrayBuffer>[]
  stop: () => Promise<void>
}

export type StartMic = (
  targetRate: number,
  onPcm: (pcm: Int16Array<ArrayBuffer>) => void,
  /** Asked per batch: true while the caller's own sound must not be sent. */
  muted: () => boolean,
  /** The caller's own level (RMS) per batch, muted or not, so speech over the agent can be
   * noticed (barge-in). */
  onLevel?: (rms: number) => void,
) => Promise<Mic>

/** One batch of line audio, as 16-bit PCM. */
export interface LineBatch {
  /** What goes down the line now: the caller (silence while muted) with the line noise. */
  sent: Int16Array<ArrayBuffer>
  /** The caller with the same noise on top, kept while muted for a barge-in to send. */
  caller: Int16Array<ArrayBuffer>
}

/**
 * One batch as it goes down the line. Muting silences the caller, not the line, so the noise
 * carries on through the agent's turn as it would on a real call. The noise is drawn once,
 * so what a barge-in sends later has exactly the noise the line had then.
 */
export function lineBatch(
  samples: Float32Array<ArrayBuffer>,
  muted: boolean,
  noise: NoiseMixer,
): LineBatch {
  const noiseOnly = noise.mix(new Float32Array(samples.length))
  const caller = floatToInt16(samples.map((sample, i) => sample + noiseOnly[i]))
  return { sent: muted ? floatToInt16(noiseOnly) : caller, caller }
}

/**
 * Captures the microphone, downsamples it to `targetRate`, mixes in any background line noise
 * and hands over 16-bit PCM batches.
 * Echo cancellation and noise suppression are on, so the agent's own voice from the
 * speakers is mostly removed before it reaches speech recognition. If any setup step
 * fails, everything already acquired is released before the error is rethrown.
 */
export const startMic: StartMic = async (targetRate, onPcm, muted, onLevel) => {
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
    let recent: Int16Array<ArrayBuffer>[] = []

    capture.port.onmessage = (event: MessageEvent<Float32Array>) => {
      pending.push(...downsampler.process(event.data))
      if (pending.length >= batchSize) {
        const batch = Float32Array.from(pending)
        onLevel?.(rmsOf(batch))
        const isMuted = muted()
        const { sent, caller } = lineBatch(batch, isMuted, noise)
        onPcm(sent)
        // With the line noise on, so a barge-in sounds like the rest of the call.
        if (isMuted) recent = [...recent, caller].slice(-RECENT_BATCHES)
        else recent = []
        pending = []
      }
    }
    source.connect(capture)

    return {
      setNoise: (level) => {
        noise.level = level
      },
      takeRecent: () => {
        const taken = recent
        recent = []
        return taken
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

export function rmsOf(samples: Float32Array): number {
  let sum = 0
  for (const sample of samples) sum += sample * sample
  return samples.length ? Math.sqrt(sum / samples.length) : 0
}
