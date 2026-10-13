import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server forwards /ws/* to the backend, so the browser talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // 127.0.0.1, not localhost: uvicorn binds IPv4 and localhost may resolve to ::1.
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
