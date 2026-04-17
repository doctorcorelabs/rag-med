import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Expose VITE_ prefixed env vars to browser
  // Set in .env.development / .env.production
})
