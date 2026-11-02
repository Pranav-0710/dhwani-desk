import { describe, expect, it } from 'vitest'
import { CallClient, type CallSocket } from './call-client'
import { callErrorText } from './error-text'

/** The error a real CallClient reports, so these tests fail if its wording moves away. */
async function clientError(act: (socket: CallSocket) => void, micError?: Error): Promise<string> {
  const errors: string[] = []
  const socket: CallSocket = {
    binaryType: 'arraybuffer',
    readyState: 1,
    onmessage: null,
    onclose: null,
    onerror: null,
    send: () => {},
    close: () => {},
  }
  const client = new CallClient(
    (event) => {
      if (event.type === 'error') errors.push(event.message)
    },
    {
      openSocket: () => socket,
      startMic: () => Promise.reject(micError ?? new Error('unused')),
      createPlayer: () => ({ start() {}, push() {}, remaining: () => 0, stop() {}, close: () => Promise.resolve() }),
      now: () => 0,
    },
  )
  client.start('ws://test')
  act(socket)
  await Promise.resolve()
  await Promise.resolve()
  return errors[0]
}

const send = (socket: CallSocket, message: object) => socket.onmessage?.({ data: JSON.stringify(message) })

describe('callErrorText', () => {
  it.each([
    ['NotAllowedError', 'Permission denied', 'blocked'],
    ['NotFoundError', 'Requested device not found', 'No microphone'],
    ['NotReadableError', 'Could not start audio source', 'Another app'],
    ['OverconstrainedError', 'No match', 'could not start'],
  ])('says what to do about a microphone failure: %s', async (name, detail, expected) => {
    const failure = new DOMException(detail, name)
    const message = await clientError((socket) => send(socket, { type: 'connected' }), failure)

    expect(message).toContain(name)
    expect(callErrorText(message)).toContain(expected)
    expect(callErrorText(message)).not.toContain(name)
  })

  it('offers a retry when the line drops', async () => {
    const message = await clientError((socket) => socket.onerror?.())

    expect(callErrorText(message)).toBe('The line dropped. Press Call to try again.')
  })

  it("says the call goes on when the agent's voice fails, without the service's detail", async () => {
    const message = await clientError((socket) => send(socket, { type: 'tts_error', message: 'HTTP 503' }))

    expect(callErrorText(message)).toContain('still on')
    expect(callErrorText(message)).not.toContain('503')
  })

  it("keeps the speech service's close reason out of the caller's view", async () => {
    const message = await clientError((socket) => send(socket, { type: 'closed', reason: 'quota' }))

    expect(callErrorText(message)).toContain('Press Call')
    expect(callErrorText(message)).not.toContain('quota')
  })

  it("shows the backend's own messages as they came", () => {
    const busy = 'The line is busy; please try again in a minute'

    expect(callErrorText(busy)).toBe(busy)
  })
})
