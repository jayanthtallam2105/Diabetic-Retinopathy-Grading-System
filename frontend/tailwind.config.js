/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        dr: {
          none: '#22c55e',
          mild: '#a3e635',
          moderate: '#facc15',
          severe: '#fb923c',
          proliferative: '#ef4444',
        },
      },
    },
  },
  plugins: [],
};

