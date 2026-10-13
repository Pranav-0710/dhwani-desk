import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server forwards /ws/* to the backend, so the browser talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
