import { describe, expect, it } from 'vitest'
import { Downsampler, floatToInt16, int16BytesToFloat } from './pcm'

/** Output samples to skip while the filter fills (63 taps at a 6:1 ratio ≈ 11 outputs). */
const WARM_UP = 16

function tone(frequency: number, rate: number, seconds: number): Float32Array {
  return Float32Array.from({ length: rate * seconds }, (_, i) =>
    Math.sin((2 * Math.PI * frequency * i) / rate),
  )
}

function rms(samples: Float32Array): number {
  const settled = samples.subarray(WARM_UP)
  return Math.sqrt(settled.reduce((sum, s) => sum + s * s, 0) / settled.length)
}

describe('Downsampler', () => {
  it('turns 48 kHz into 8 kHz with the right sample count and keeps DC level', () => {
    const out = new Downsampler(48000, 8000).process(new Float32Array(4800).fill(0.5))

    expect(out.length).toBe(800)
    expect(out.subarray(WARM_UP).every((s) => Math.abs(s - 0.5) < 1e-3)).toBe(true)
  })

  it('passes speech-band audio almost unchanged', () => {
    const out = new Downsampler(48000, 8000).process(tone(1000, 48000, 1))

    expect(rms(out)).toBeGreaterThan(0.707 * 0.9) // within 1 dB of the input
  })

  it('blocks audio above the telephone band instead of aliasing it', () => {
    // A 6 kHz tone would fold down to 2 kHz at 8 kHz without filtering.
    const out = new Downsampler(48000, 8000).process(tone(6000, 48000, 1))

    expect(rms(out)).toBeLessThan(0.707 * 0.1) // at least 20 dB down
  })

  it('gives the same output whatever the chunk sizes', () => {
    const input = Float32Array.from({ length: 4410 }, (_, i) => Math.sin(i / 7))
    const whole = new Downsampler(44100, 8000).process(input)

    const chunked = new Downsampler(44100, 8000)
    const parts: number[] = []
    for (let start = 0; start < input.length; start += 128) {
      parts.push(...chunked.process(input.subarray(start, start + 128)))
    }

    expect(parts.length).toBe(whole.length)
    parts.forEach((s, i) => expect(s).toBeCloseTo(whole[i], 5))
  })

  it('keeps the long-run rate exact for a non-integer ratio', () => {
    const out = new Downsampler(44100, 8000).process(new Float32Array(44100))

    expect(Math.abs(out.length - 8000)).toBeLessThanOrEqual(1)
  })

  it('refuses to upsample', () => {
    expect(() => new Downsampler(8000, 16000)).toThrow()
  })
})

describe('PCM conversion', () => {
  it('clamps and scales floats to int16', () => {
    expect(Array.from(floatToInt16(Float32Array.from([0, 1, -1, 2, -2])))).toEqual([
      0, 32767, -32768, 32767, -32768,
    ])
  })

  it('reads little-endian int16 bytes as floats', () => {
    const bytes = new Uint8Array([0x00, 0x40, 0x00, 0xc0]).buffer // 16384, -16384

    expect(Array.from(int16BytesToFloat(bytes))).toEqual([0.5, -0.5])
  })
})
