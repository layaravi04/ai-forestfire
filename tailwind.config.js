/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: "#0B0F19",
        cardBg: "#111827",
        panelBg: "#1F2937",
        fireRed: "#EF4444",
        smokeOrange: "#F97316",
        warningYellow: "#EAB308",
        safeGreen: "#10B981"
      }
    },
  },
  plugins: [],
}
