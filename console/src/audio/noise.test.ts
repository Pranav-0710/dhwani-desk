import { describe, expect, it } from 'vitest'
import { NoiseMixer, seededRandom, type NoiseLevel } from './noise'

const RATE = 8000

function rms(samples: Float32Array): number {
  return Math.sqrt(samples.reduce((sum, s) => sum + s * s, 0) / samples.length)
}

function noiseAlone(level: NoiseLevel, seconds = 10, seed = 1): Float32Array {
  const mixer = new NoiseMixer(seededRandom(seed))
  mixer.level = level
  return mixer.mix(new Float32Array(RATE * seconds))
}

describe('NoiseMixer', () => {
  it('leaves the audio untouched when off', () => {
    const speech = Float32Array.from({ length: 400 }, (_, i) => Math.sin(i / 5))

    expect(new NoiseMixer().mix(speech)).toBe(speech)
  })

  it('adds noise at about -34 dBFS when light and -24 dBFS when heavy', () => {
    expect(rms(noiseAlone('light'))).toBeCloseTo(0.02, 2)
    expect(rms(noiseAlone('heavy'))).toBeCloseTo(0.063, 2)
  })

  it('adds the noise to the speech rather than replacing it', () => {
    const speech = new Float32Array(RATE).fill(0.25)
    const mixer = new NoiseMixer(seededRandom(7))
    mixer.level = 'light'

    const mixed = mixer.mix(speech)
    const mean = mixed.reduce((sum, s) => sum + s, 0) / mixed.length

    expect(mean).toBeCloseTo(0.25, 1)
    expect(Array.from(speech).every((s) => s === 0.25)).toBe(true) // input not changed
  })

  it('is pink, not white: neighbouring samples move together', () => {
    const noise = noiseAlone('heavy')
    let together = 0
    for (let i = 1; i < noise.length; i++) together += noise[i] * noise[i - 1]
    const correlation = together / (noise.length - 1) / rms(noise) ** 2

    expect(correlation).toBeGreaterThan(0.5) // white noise would be about 0
  })

  it('follows a level change from the next batch', () => {
    const mixer = new NoiseMixer(seededRandom(3))
    mixer.level = 'heavy'
    const first = mixer.mix(new Float32Array(320))
    mixer.level = 'off'
    const quiet = new Float32Array(320)

    expect(first.some((s) => s !== 0)).toBe(true)
    expect(mixer.mix(quiet)).toBe(quiet)
  })

  it('gives the same noise for the same seed', () => {
    expect(Array.from(noiseAlone('light', 1, 42))).toEqual(Array.from(noiseAlone('light', 1, 42)))
  })
})
