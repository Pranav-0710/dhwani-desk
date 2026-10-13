/** Low-pass cutoff as a fraction of the output rate (3.6 kHz for 8 kHz telephony audio). */
const CUTOFF_FRACTION = 0.45
const TAPS = 63

/**
 * Downsamples a live mono stream (e.g. 48 kHz mic) to a lower rate (8 kHz for Prisma).
 *
 * A windowed-sinc FIR low-pass removes everything above the new band before samples are
 * dropped, so sibilants and fricatives don't alias down into the band Prisma listens to.
 * Filter state and decimation phase carry across calls: feeding audio in any chunk sizes
 * gives the same output, and non-integer ratios (44.1 kHz → 8 kHz) keep the exact rate.
 */
export class Downsampler {
  private readonly step: number
  private readonly taps: Float32Array
  private readonly history: Float32Array
  private position = 0
  private phase = 0

  constructor(inputRate: number, outputRate: number) {
    if (outputRate > inputRate) {
      throw new Error(`Cannot downsample ${inputRate} Hz to ${outputRate} Hz`)
    }
    this.step = inputRate / outputRate
    this.taps = lowPassTaps((CUTOFF_FRACTION * outputRate) / inputRate, TAPS)
    this.history = new Float32Array(TAPS)
  }

  process(input: Float32Array): Float32Array<ArrayBuffer> {
    const out: number[] = []
    for (const sample of input) {
      this.history[this.position] = sample
      this.position = (this.position + 1) % TAPS
      this.phase += 1
      if (this.phase >= this.step) {
        this.phase -= this.step
        let sum = 0
        for (let k = 0; k < TAPS; k++) {
          sum += this.taps[k] * this.history[(this.position + k) % TAPS]
        }
        out.push(sum)
      }
    }
    return Float32Array.from(out)
  }
}

/** Hamming-windowed sinc low-pass, normalised to unity gain at DC. `cutoff` is cycles/sample. */
function lowPassTaps(cutoff: number, count: number): Float32Array {
  const taps = new Float32Array(count)
  const middle = (count - 1) / 2
  let total = 0
  for (let n = 0; n < count; n++) {
    const x = n - middle
    const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x)
    const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (count - 1))
    taps[n] = sinc * window
    total += taps[n]
  }
  return taps.map((tap) => tap / total)
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
