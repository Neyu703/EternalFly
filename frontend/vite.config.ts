import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // three.js's core is one ~740 kB module that can't be split further; every other
    // chunk stays well below this, so the warning still catches real growth.
    chunkSizeWarningLimit: 800,
    rolldownOptions: {
      output: {
        // Libraries in their own chunks, so the size of each (and of the app code) is
        // visible and bounded on its own, and they stay cached across app updates.
        codeSplitting: {
          groups: [
            { name: 'three', test: /node_modules[\\/]three[\\/]/ },
            { name: 'react-three', test: /node_modules[\\/]@react-three[\\/]/ },
            { name: 'vendor', test: /node_modules[\\/]/ },
          ],
        },
      },
    },
  },
})
