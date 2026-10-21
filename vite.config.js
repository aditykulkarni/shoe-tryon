import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// HTTPS with a self-signed certificate so phones on the same Wi-Fi can open
// the camera from the dev server.
export default defineConfig({
  base: './',
  plugins: [basicSsl()],
});
