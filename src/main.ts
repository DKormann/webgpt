import "./style.css";

const app = document.querySelector<HTMLDivElement>("#app");

if (app) {
  app.innerHTML = `
    <h1>Bun + Vite</h1>
    <p>Dev server has live reload. Build output goes to <code>./docs</code>.</p>
  `;
}
