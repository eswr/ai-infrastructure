import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

export default defineConfig({
  plugins: [solid()],
  build: { target: "esnext" },
  server: {
    // `npm run dev` only for UI development; the demo path is
    // `make console`, which serves the built SPA from server.mjs.
    proxy: { "/api": "http://127.0.0.1:8099" },
  },
});
