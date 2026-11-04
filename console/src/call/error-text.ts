/**
 * What the phone page tells a caller about an error from CallClient. The client's own messages
 * name browser errors and services; these say what happened and what to do next. Anything not
 * listed here is shown as it came: the backend's messages are already written for a caller.
 */
export function callErrorText(message: string): string {
  if (message.startsWith('Microphone unavailable')) {
    if (/NotAllowedError|SecurityError/.test(message)) {
      return "The microphone is blocked. Allow it for this page in your browser's address bar, then call again."
    }
    if (message.includes('NotFoundError')) return 'No microphone was found. Connect one, then call again.'
    if (message.includes('NotReadableError')) {
      return 'Another app is using the microphone. Close it, then call again.'
    }
    return 'The microphone could not start. Check it, then call again.'
  }
  if (message === 'Lost connection to the backend') {
    return 'The line dropped. Press Call to try again.'
  }
  if (message.startsWith('Agent voice failed')) {
    return "The agent's voice failed for that line, so read it below. The call is still on."
  }
  if (message.startsWith('Speech service ended the call')) {
    return 'The speech service ended the call. Press Call to try again.'
  }
  return message
}
