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
    const pcm = lineBatch(speech(), false, mixer('off')).sent

    expect(pcm.length).toBe(320)
    expect(pcm[3]).toBe(Math.round(Math.fround(0.5 * Math.sin(1)) * 0x7fff))
  })

  it('sends silence while muted on a quiet line', () => {
    expect(lineBatch(speech(), true, mixer('off')).sent.every((s) => s === 0)).toBe(true)
  })

  it('keeps the line noise going while the caller is muted', () => {
    const muted = lineBatch(speech(), true, mixer('heavy')).sent
    const noiseOnly = lineBatch(new Float32Array(320), false, mixer('heavy')).sent

    expect(muted.some((s) => s !== 0)).toBe(true)
    expect(Array.from(muted)).toEqual(Array.from(noiseOnly)) // the noise, and nothing of the caller
  })

  it('clamps speech near full scale once the noise is added', () => {
    const loud = new Float32Array(320).fill(0.99)
    const pcm = lineBatch(loud, false, mixer('heavy')).sent

    expect(Math.max(...pcm)).toBe(32767)
    expect(Math.min(...pcm)).toBeGreaterThanOrEqual(-32768)
  })

  it('keeps the caller with the same noise while muted, for a barge-in', () => {
    const { sent, caller } = lineBatch(speech(), true, mixer('heavy'))
    const unmuted = lineBatch(speech(), false, mixer('heavy')).sent

    expect(Array.from(caller)).toEqual(Array.from(unmuted)) // the caller and the noise
    expect(Array.from(caller)).not.toEqual(Array.from(sent))
  })

  it('keeps the caller clean on a quiet line', () => {
    const { caller } = lineBatch(speech(), true, mixer('off'))

    expect(Array.from(caller)).toEqual(Array.from(lineBatch(speech(), false, mixer('off')).sent))
  })
})
