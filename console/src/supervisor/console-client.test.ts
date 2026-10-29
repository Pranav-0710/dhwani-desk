import { afterEach, describe, expect, it, vi } from 'vitest'
import { ConsoleClient, type ConsoleSocket, type ConsoleStatus } from './console-client'
import type { ServerMessage } from './protocol'

class FakeSocket implements ConsoleSocket {
  readyState = 0
  sent: unknown[] = []
  onopen: (() => void) | null = null
  onmessage: ((message: { data: unknown }) => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  send(data: string) {
    this.sent.push(JSON.parse(data))
  }
  close() {}
  open() {
    this.readyState = 1
    this.onopen?.()
  }
  receive(message: object) {
    this.onmessage?.({ data: JSON.stringify(message) })
  }
}

function connect() {
  const socket = new FakeSocket()
  const messages: ServerMessage[] = []
  const statuses: [ConsoleStatus, string | undefined][] = []
  const client = new ConsoleClient(
    (m) => messages.push(m),
    (s, detail) => statuses.push([s, detail]),
    () => socket,
  )
  client.connect('secret', 'ws://test/ws/console')
  socket.open()
  return { socket, client, messages, statuses }
}

afterEach(() => vi.useRealTimers())

describe('ConsoleClient', () => {
  it('says hello with the token, then reports live once the calls arrive', () => {
    const { socket, statuses, messages } = connect()
    expect(socket.sent).toEqual([{ type: 'hello', token: 'secret' }])

    socket.receive({ type: 'calls', calls: [], thresholds: {} })

    expect(statuses.map(([s]) => s)).toEqual(['connecting', 'connected'])
    expect(messages[0].type).toBe('calls')
  })

  it('reports a refused token with the server reason', () => {
    const { socket, statuses } = connect()

    socket.receive({ type: 'error', message: 'Not allowed' })
    socket.onclose?.({ code: 1008 })

    expect(statuses.at(-1)).toEqual(['refused', 'Not allowed'])
  })

  it("settles each command by the server's answer to its id", async () => {
    const { socket, client } = connect()

    const takeOver = client.send('take_over', 'c1')
    const say = client.send('say', 'c1', 'ನಮಸ್ಕಾರ')
    expect(socket.sent.slice(1)).toEqual([
      { type: 'take_over', call_id: 'c1', id: 1 },
      { type: 'say', call_id: 'c1', id: 2, text: 'ನಮಸ್ಕಾರ' },
    ])

    socket.receive({ type: 'error', message: 'Take the call over before speaking on it', command: 'say', call_id: 'c1', id: 2 })
    socket.receive({ type: 'done', command: 'take_over', call_id: 'c1', id: 1 })

    await expect(takeOver).resolves.toBeUndefined()
    await expect(say).rejects.toThrow('Take the call over before speaking on it')
  })

  it('gives up on a command the server never answers', async () => {
    vi.useFakeTimers()
    const { client } = connect()

    const command = client.send('hand_back', 'c1')
    vi.advanceTimersByTime(12_000)

    await expect(command).rejects.toThrow('did not answer in time')
  })

  it('fails waiting commands when the connection drops, and refuses new ones', async () => {
    const { socket, client } = connect()

    const command = client.send('take_over', 'c1')
    socket.readyState = 3
    socket.onclose?.({ code: 1006 })

    await expect(command).rejects.toThrow('lost its connection')
    await expect(client.send('take_over', 'c1')).rejects.toThrow('not connected')
  })

  it("drops a closed connection's handlers, so it cannot reach the next one", () => {
    const { socket, client, statuses, messages } = connect()
    client.close()

    socket.receive({ type: 'calls', calls: [], thresholds: {} })
    socket.onclose?.({ code: 1006 })

    expect(statuses.map(([s]) => s)).toEqual(['connecting'])
    expect(messages).toEqual([])
  })

  it('passes on why the server ended the session', () => {
    const { socket, statuses } = connect()

    socket.receive({ type: 'error', message: 'Fell too far behind; reconnect' })
    socket.onclose?.({ code: 1000 })

    expect(statuses.at(-1)).toEqual(['closed', 'Fell too far behind; reconnect'])
  })
})
