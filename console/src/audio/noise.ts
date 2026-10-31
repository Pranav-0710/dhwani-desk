/**
 * Background noise mixed into the caller's audio, so the demo hears what a phone line in a
 * busy street sounds like, not a quiet laptop microphone.
 *
 * Pink noise (equal energy per octave, so it rumbles more than it hisses, like traffic and a
 * crowd) is added after the browser's own noise suppression and after downsampling, so speech
 * recognition gets all of it.
 */

export type NoiseLevel = 'off' | 'light' | 'heavy'

export const NOISE_LEVELS: readonly NoiseLevel[] = ['off', 'light', 'heavy']

/** RMS of the added noise: about −34 dBFS (a quiet street) and −24 dBFS (a busy one). */
const NOISE_RMS: Record<NoiseLevel, number> = { off: 0, light: 0.02, heavy: 0.063 }

/** RMS of `pink()` for white input of unit variance (2.95–3.00 over a minute of samples). */
const PINK_RMS = 2.97

export class NoiseMixer {
  level: NoiseLevel = 'off'
  private readonly random: () => number
  private b0 = 0
  private b1 = 0
  private b2 = 0

  constructor(random: () => number = Math.random) {
    this.random = random
  }

  /** The samples with this level's noise added; unchanged (the same array) when off. */
  mix(samples: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> {
    const gain = NOISE_RMS[this.level] / PINK_RMS
    if (gain === 0) return samples
    const out = new Float32Array(samples.length)
    for (let i = 0; i < samples.length; i++) {
      out[i] = samples[i] + gain * this.pink()
    }
    return out
  }

  /** Paul Kellet's economy pink filter over white Gaussian noise. */
  private pink(): number {
    const white = gaussian(this.random)
    this.b0 = 0.99765 * this.b0 + white * 0.099046
    this.b1 = 0.963 * this.b1 + white * 0.2965164
    this.b2 = 0.57 * this.b2 + white * 1.0526913
    return this.b0 + this.b1 + this.b2 + white * 0.1848
  }
}

/** A standard normal sample (Box–Muller). */
function gaussian(random: () => number): number {
  const u = 1 - random() // never 0, so the log is finite
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random())
}

/** A small seeded generator (mulberry32), so tests get the same noise every run. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
