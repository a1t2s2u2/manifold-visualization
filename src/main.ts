import "./style.css";
import { SceneManager } from "./scene";
import { manifolds } from "./manifolds";
import type { ManifoldDefinition } from "./types";

class App {
  private scene: SceneManager;
  private currentManifold: ManifoldDefinition;
  private currentParams: Record<string, number | string | boolean> = {};

  constructor() {
    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    this.scene = new SceneManager(canvas);

    this.currentManifold = manifolds[0]!;
    this.initParams(this.currentManifold);

    this.renderManifoldList();
    this.renderParamsPanel();
    this.renderInfoPanel();
    this.generateAndDisplay();

    this.scene.onFps((fps) => {
      const el = document.getElementById("fps-counter");
      if (el) el.textContent = `${fps} FPS`;
    });

    this.updateVertexCount();
  }

  private initParams(manifold: ManifoldDefinition): void {
    this.currentParams = {};
    for (const p of manifold.defaultParams) {
      this.currentParams[p.key] = p.value;
    }
  }

  private renderManifoldList(): void {
    const container = document.getElementById("manifold-list")!;
    container.innerHTML = "";

    for (const m of manifolds) {
      const btn = document.createElement("button");
      btn.className =
        "manifold-btn" + (m.id === this.currentManifold.id ? " active" : "");
      btn.innerHTML = `
        <span class="manifold-name">${m.info.name}</span>
        <span class="manifold-math">${m.info.mathSymbol}</span>
      `;
      btn.addEventListener("click", () => {
        this.currentManifold = m;
        this.initParams(m);
        this.renderManifoldList();
        this.renderParamsPanel();
        this.renderInfoPanel();
        this.generateAndDisplay();
      });
      container.appendChild(btn);
    }
  }

  private renderParamsPanel(): void {
    const container = document.getElementById("params-panel")!;
    container.innerHTML = "";

    if (this.currentManifold.defaultParams.length === 0) return;

    const title = document.createElement("h3");
    title.textContent = "パラメータ";
    container.appendChild(title);

    for (const param of this.currentManifold.defaultParams) {
      const group = document.createElement("div");
      group.className = "param-group";

      if (param.type === "range") {
        const label = document.createElement("label");
        const valueSpan = document.createElement("span");
        valueSpan.className = "param-value";
        valueSpan.textContent = String(this.currentParams[param.key]);
        label.textContent = param.label;
        label.appendChild(valueSpan);

        const input = document.createElement("input");
        input.type = "range";
        input.min = String(param.min ?? 0);
        input.max = String(param.max ?? 100);
        input.step = String(param.step ?? 1);
        input.value = String(this.currentParams[param.key]);

        input.addEventListener("input", () => {
          const val = parseFloat(input.value);
          this.currentParams[param.key] = val;
          valueSpan.textContent = String(val);
        });
        input.addEventListener("change", () => {
          this.generateAndDisplay();
        });

        group.appendChild(label);
        group.appendChild(input);
      } else if (param.type === "checkbox") {
        const wrapper = document.createElement("div");
        wrapper.className = "checkbox-group";

        const input = document.createElement("input");
        input.type = "checkbox";
        input.id = `param-${param.key}`;
        input.checked = this.currentParams[param.key] as boolean;

        input.addEventListener("change", () => {
          this.currentParams[param.key] = input.checked;
          this.generateAndDisplay();
        });

        const label = document.createElement("label");
        label.htmlFor = `param-${param.key}`;
        label.textContent = param.label;

        wrapper.appendChild(input);
        wrapper.appendChild(label);
        group.appendChild(wrapper);
      } else if (param.type === "select") {
        const label = document.createElement("label");
        label.textContent = param.label;

        const select = document.createElement("select");
        for (const opt of param.options ?? []) {
          const option = document.createElement("option");
          option.value = opt.value;
          option.textContent = opt.label;
          if (opt.value === this.currentParams[param.key]) {
            option.selected = true;
          }
          select.appendChild(option);
        }

        select.addEventListener("change", () => {
          this.currentParams[param.key] = select.value;
          this.generateAndDisplay();
        });

        group.appendChild(label);
        group.appendChild(select);
      }

      container.appendChild(group);
    }
  }

  private renderInfoPanel(): void {
    const container = document.getElementById("info-panel")!;
    const info = this.currentManifold.info;

    container.innerHTML = `
      <h3>情報</h3>
      <div class="info-content">
        <p>${info.description}</p>
        <p><span class="math">dim = ${info.dimension}</span></p>
        <p><strong>性質:</strong></p>
        ${info.properties.map((p) => `<p>・${p}</p>`).join("")}
      </div>
    `;
  }

  private generateAndDisplay(): void {
    // Enforce k <= n for Stiefel/Grassmann
    if (
      this.currentManifold.id === "stiefel" ||
      this.currentManifold.id === "grassmann"
    ) {
      const k = this.currentParams["k"] as number;
      const n = this.currentParams["n"] as number;
      if (k > n) {
        // k を上げた場合は n を引き上げ、n を下げた場合は k を引き下げ
        const nParam = this.currentManifold.defaultParams.find(p => p.key === "n");
        const nMax = (nParam?.max as number) ?? 8;
        if (k <= nMax) {
          this.currentParams["n"] = k;
        } else {
          this.currentParams["k"] = n;
        }
        this.renderParamsPanel();
      }
    }

    const obj = this.currentManifold.generate(this.currentParams);
    this.scene.setObject(obj);
    this.updateVertexCount();
  }

  private updateVertexCount(): void {
    requestAnimationFrame(() => {
      const el = document.getElementById("vertex-count");
      if (el) {
        const count = this.scene.getVertexCount();
        el.textContent = `${count.toLocaleString()} vertices`;
      }
    });
  }
}

new App();
