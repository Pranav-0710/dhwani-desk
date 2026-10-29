import { describe, expect, it } from 'vitest'
import { lineBatch } from './mic'
import { NoiseMixer, seededRandom } from './noise'

const speech = () => Float32Array.from({ length: 320 }, (_, i) => 0.5 * Math.sin(i / 3))

function mixer(level: 'off' | 'heavy'): NoiseMixer {
  const noise = new NoiseMixer(seededRandom(5))
  noise.level = level
  return noise
}

describe('lineBatch', () => {
  it('sends the caller as 16-bit PCM when the line is quiet', () => {
    const pcm = lineBatch(speech(), false, mixer('off'))

    expect(pcm.length).toBe(320)
    expect(pcm[3]).toBe(Math.round(Math.fround(0.5 * Math.sin(1)) * 0x7fff))
  })

  it('sends silence while muted on a quiet line', () => {
    expect(lineBatch(speech(), true, mixer('off')).every((s) => s === 0)).toBe(true)
  })

  it('keeps the line noise going while the caller is muted', () => {
    const muted = lineBatch(speech(), true, mixer('heavy'))
    const noiseOnly = lineBatch(new Float32Array(320), false, mixer('heavy'))

    expect(muted.some((s) => s !== 0)).toBe(true)
    expect(Array.from(muted)).toEqual(Array.from(noiseOnly)) // the noise, and nothing of the caller
  })

  it('clamps speech near full scale once the noise is added', () => {
    const loud = new Float32Array(320).fill(0.99)
    const pcm = lineBatch(loud, false, mixer('heavy'))

    expect(Math.max(...pcm)).toBe(32767)
    expect(Math.min(...pcm)).toBeGreaterThanOrEqual(-32768)
  })
})
