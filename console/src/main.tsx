import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Supervisor from './supervisor/Supervisor.tsx'

// Two pages from one app: the caller's phone, and the officers' console at /supervisor.
const Page = window.location.pathname.startsWith('/supervisor') ? Supervisor : App

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Page />
  </StrictMode>,
)
