/**
 * Downsamples a live mono stream (e.g. 48 kHz mic) to a lower rate (8 kHz for Prisma).
 *
 * Each output sample is the average of the input samples in its window — a box filter that
 * removes most of the energy that would otherwise alias into the telephone band. Window
 * boundaries carry across calls, so feeding audio in any chunk sizes gives the same output.
 */
export class Downsampler {
  private readonly step: number
  private phase = 0
  private sum = 0
  private count = 0

  constructor(inputRate: number, outputRate: number) {
    if (outputRate > inputRate) {
      throw new Error(`Cannot downsample ${inputRate} Hz to ${outputRate} Hz`)
    }
    this.step = inputRate / outputRate
  }

  process(input: Float32Array): Float32Array<ArrayBuffer> {
    const out: number[] = []
    for (const sample of input) {
      this.sum += sample
      this.count += 1
      this.phase += 1
      if (this.phase >= this.step) {
        out.push(this.sum / this.count)
        this.sum = 0
        this.count = 0
        this.phase -= this.step
      }
    }
    return Float32Array.from(out)
  }
}

/** Float samples in [-1, 1] to 16-bit signed PCM, clamping anything louder. */
export function floatToInt16(samples: Float32Array): Int16Array<ArrayBuffer> {
  const out = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff)
  }
  return out
}

/** 16-bit signed little-endian PCM bytes to float samples for Web Audio. */
export function int16BytesToFloat(bytes: ArrayBuffer): Float32Array<ArrayBuffer> {
  const view = new DataView(bytes)
  const out = new Float32Array(Math.floor(bytes.byteLength / 2))
  for (let i = 0; i < out.length; i++) {
    out[i] = view.getInt16(i * 2, true) / 0x8000
  }
  return out
}
