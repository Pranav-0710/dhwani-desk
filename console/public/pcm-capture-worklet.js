// Runs on the audio thread: copies each 128-sample block of the mic's first channel to the
// main thread, which downsamples it and sends it to the backend.
class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (channel) {
      this.port.postMessage(channel.slice(0))
    }
    return true
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor)
