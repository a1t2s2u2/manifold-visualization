import "./style.css";
import { SceneManager } from "./scene";
import { loadMnistStiefelData } from "./dataLoader";
import { buildVisualization, removeColorBar } from "./visualization";
import { drawLossChart, removeLossChart } from "./lossChart";
import type { MnistStiefelData, LandscapeMode } from "./types";

class App {
  private scene: SceneManager;
  private data: MnistStiefelData | null = null;
  private showPath = true;
  private landscapeMode: LandscapeMode = "local";

  constructor() {
    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    this.scene = new SceneManager(canvas);

    this.scene.onFps((fps) => {
      const el = document.getElementById("fps-counter");
      if (el) el.textContent = `${fps} FPS`;
    });

    this.loadAndRender();
  }

  private async loadAndRender(): Promise<void> {
    try {
      this.data = await loadMnistStiefelData();
      this.hideLoading();
      this.renderModelInfo();
      this.renderControls();
      this.renderInfoPanel();
      this.updateVisualization();
      drawLossChart(this.data);
    } catch (e) {
      this.hideLoading();
      const info = document.getElementById("model-info")!;
      info.innerHTML = `
        <div class="error-message">
          <p>データの読み込みに失敗しました。</p>
          <p class="error-detail">python train.py を実行してデータを生成してください。</p>
        </div>
      `;
    }
  }

  private hideLoading(): void {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.style.display = "none";
  }

  private renderModelInfo(): void {
    if (!this.data) return;
    const container = document.getElementById("model-info")!;
    const m = this.data.metadata;
    container.innerHTML = `
      <h3>モデル</h3>
      <div class="model-details">
        <p class="model-formula">${m.model}</p>
        <div class="model-stats">
          <div class="stat">
            <span class="stat-label">Accuracy</span>
            <span class="stat-value">${(m.final_accuracy * 100).toFixed(1)}%</span>
          </div>
          <div class="stat">
            <span class="stat-label">Loss</span>
            <span class="stat-value">${m.final_loss.toFixed(4)}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Epochs</span>
            <span class="stat-value">${m.epochs}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Learning Rate</span>
            <span class="stat-value">${m.learning_rate}</span>
          </div>
        </div>
      </div>
    `;
  }

  private renderControls(): void {
    const container = document.getElementById("controls-panel")!;
    container.innerHTML = "";

    const title = document.createElement("h3");
    title.textContent = "コントロール";
    container.appendChild(title);

    // Landscape mode select
    const modeGroup = document.createElement("div");
    modeGroup.className = "param-group";
    const modeLabel = document.createElement("label");
    modeLabel.textContent = "ランドスケープ";
    modeGroup.appendChild(modeLabel);

    const modeSelect = document.createElement("select");
    const options: { value: LandscapeMode; label: string }[] = [
      { value: "local", label: "パス周辺 (局所)" },
      { value: "global", label: "全体 (グローバル)" },
    ];
    for (const opt of options) {
      const el = document.createElement("option");
      el.value = opt.value;
      el.textContent = opt.label;
      if (opt.value === this.landscapeMode) el.selected = true;
      modeSelect.appendChild(el);
    }
    modeSelect.addEventListener("change", () => {
      this.landscapeMode = modeSelect.value as LandscapeMode;
      this.updateVisualization();
    });
    modeGroup.appendChild(modeSelect);
    container.appendChild(modeGroup);

    // Show path toggle
    const pathGroup = document.createElement("div");
    pathGroup.className = "checkbox-group";
    const pathInput = document.createElement("input");
    pathInput.type = "checkbox";
    pathInput.id = "show-path";
    pathInput.checked = this.showPath;
    pathInput.addEventListener("change", () => {
      this.showPath = pathInput.checked;
      this.updateVisualization();
    });
    const pathLabel = document.createElement("label");
    pathLabel.htmlFor = "show-path";
    pathLabel.textContent = "最適化パス表示";
    pathGroup.appendChild(pathInput);
    pathGroup.appendChild(pathLabel);
    container.appendChild(pathGroup);
  }

  private renderInfoPanel(): void {
    if (!this.data) return;
    const container = document.getElementById("info-panel")!;
    const view = this.data[this.landscapeMode];
    const variance = view.pca_explained_variance;

    container.innerHTML = `
      <h3>情報</h3>
      <div class="info-content">
        <p>MNIST手書き数字の10クラス分類を、Stiefel多様体 St(10, 784) 上の重み行列 W で学習。Riemannian SGD により多様体上で最適化し、PCAで3次元に射影して可視化しています。</p>
        <p><strong>PCA寄与率:</strong></p>
        <p class="math">PC1: ${(variance[0]! * 100).toFixed(1)}%, PC2: ${(variance[1]! * 100).toFixed(1)}%, PC3: ${(variance[2]! * 100).toFixed(1)}%</p>
        <p><strong>可視化要素:</strong></p>
        <p>・<span style="color:#4488ff">青</span>〜<span style="color:#ff4444">赤</span> 点群: St(10,784)上の点 (loss値で色付け)</p>
        <p>・<span style="color:#ffd700">金色</span>曲線: Riemannian SGDの最適化パス</p>
        <p>・<span style="color:#ff1744">赤球</span>: 初期点, <span style="color:#00e676">緑球</span>: 最終点 (グローバルでは最終W位置のみ)</p>
        <p><strong>ランドスケープモード:</strong></p>
        <p>・<strong>パス周辺</strong>: 最適化パスのPCA方向に沿った摂動点。パスと損失面の関係が見える</p>
        <p>・<strong>全体</strong>: St(10,784)上の一様ランダム点。多様体全体の損失分布が見える (独自PCA空間)</p>
      </div>
    `;
  }

  private updateVisualization(): void {
    if (!this.data) return;
    removeLossChart();
    removeColorBar();
    const obj = buildVisualization(this.data, {
      showPath: this.showPath,
      landscapeMode: this.landscapeMode,
    });
    this.scene.setObject(obj);
    this.renderInfoPanel();
    drawLossChart(this.data);

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
