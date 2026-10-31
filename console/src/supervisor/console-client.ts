import type { ServerMessage } from './protocol'

/** WebSocket.OPEN, spelled out so the client runs without a browser. */
const SOCKET_OPEN = 1
/** The backend gives up on a command after 10 s; wait a little longer for its answer. */
const COMMAND_TIMEOUT_MS = 12_000

export type Command = 'take_over' | 'say' | 'hand_back'

export type ConsoleStatus = 'connecting' | 'connected' | 'refused' | 'closed'

/** What the client needs from a WebSocket; injectable so it can be tested. */
export interface ConsoleSocket {
  readyState: number
  onopen: (() => void) | null
  onmessage: ((message: { data: unknown }) => void) | null
  onclose: ((event: { code: number }) => void) | null
  send(data: string): void
  close(code?: number): void
}

type Pending = { resolve: () => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }

/**
 * One supervisor console connection to `/ws/console`. It says hello with the console token,
 * hands every server message to `onMessage`, and sends officer commands, each settled by the
 * server's `done` or `error` for its id.
 */
export class ConsoleClient {
  private socket: ConsoleSocket | null = null
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private readonly onMessage: (message: ServerMessage) => void
  private readonly onStatus: (status: ConsoleStatus, detail?: string) => void
  private readonly openSocket: (url: string) => ConsoleSocket

  constructor(
    onMessage: (message: ServerMessage) => void,
    onStatus: (status: ConsoleStatus, detail?: string) => void,
    openSocket: (url: string) => ConsoleSocket = (url) =>
      new WebSocket(url) as unknown as ConsoleSocket,
  ) {
    this.onMessage = onMessage
    this.onStatus = onStatus
    this.openSocket = openSocket
  }

  connect(token: string, url = consoleUrl()): void {
    this.onStatus('connecting')
    const socket = this.openSocket(url)
    // The server's reason for ending the session: a refused hello, "fell too far behind", etc.
    let reason: string | null = null
    socket.onopen = () => socket.send(JSON.stringify({ type: 'hello', token }))
    socket.onmessage = (raw) => {
      if (typeof raw.data !== 'string') return
      const message = JSON.parse(raw.data) as ServerMessage
      if (message.type === 'calls') this.onStatus('connected')
      if (message.type === 'error' && message.id === undefined && message.command === undefined) {
        reason = message.message
      }
      if (message.type === 'done' || message.type === 'error') this.settle(message)
      this.onMessage(message)
    }
    socket.onclose = (event) => {
      this.failPending('The console lost its connection')
      this.onStatus(event.code === 1008 ? 'refused' : 'closed', reason ?? undefined)
    }
    this.socket = socket
  }

  /** Close for good: the socket's handlers are dropped first, so a late close or message from
   * it cannot reach a newer connection's page. */
  close(): void {
    const socket = this.socket
    this.socket = null
    if (!socket) return
    socket.onopen = socket.onmessage = null
    socket.onclose = null
    this.failPending('The console was closed')
    socket.close(1000)
  }

  /** Resolves once the call has carried the command out; rejects with the server's reason. */
  send(command: Command, callId: string, text?: string): Promise<void> {
    const socket = this.socket
    if (socket?.readyState !== SOCKET_OPEN) {
      return Promise.reject(new Error('The console is not connected'))
    }
    const id = this.nextId++
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error('The call did not answer in time'))
      }, COMMAND_TIMEOUT_MS)
      this.pending.set(id, { resolve, reject, timer })
      socket.send(JSON.stringify({ type: command, call_id: callId, id, ...(text ? { text } : {}) }))
    })
  }

  private settle(message: Extract<ServerMessage, { type: 'done' | 'error' }>): void {
    if (typeof message.id !== 'number') return
    const pending = this.pending.get(message.id)
    if (!pending) return
    this.pending.delete(message.id)
    clearTimeout(pending.timer)
    if (message.type === 'done') pending.resolve()
    else pending.reject(new Error(message.message))
  }

  private failPending(reason: string): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer)
      pending.reject(new Error(reason))
    }
    this.pending.clear()
  }
}

function consoleUrl(): string {
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${window.location.host}/ws/console`
}
