/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
      },
      colors: {
        accent: {
          cyan: "#00FFCC",
          lime: "#84cc16",
        },
      },
      keyframes: {
        "shield-pulse": {
          "0%, 100%": { boxShadow: "inset 0 0 40px rgba(0,255,204,0.15)" },
          "50%": { boxShadow: "inset 0 0 80px rgba(0,255,204,0.25)" },
        },
      },
      animation: {
        "shield-pulse": "shield-pulse 3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
