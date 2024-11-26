/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0066cc',
          dark: '#004999',
        },
        dark: {
          DEFAULT: '#0a0a0a',
          lighter: '#1a1a1a',
        }
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px #0066cc, 0 0 10px #0066cc, 0 0 15px #0066cc' },
          '100%': { boxShadow: '0 0 10px #0066cc, 0 0 20px #0066cc, 0 0 30px #0066cc' },
        },
      },
    },
  },
  plugins: [],
};