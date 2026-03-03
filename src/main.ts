import "./style.css";
import { SceneManager } from "./scene";
import { loadMnistStiefelData } from "./dataLoader";
import { buildVisualization } from "./visualization";
import { drawLossChart, removeLossChart } from "./lossChart";
import type { MnistStiefelData } from "./types";

class App {
  private scene: SceneManager;
  private data: MnistStiefelData | null = null;
  private showPath = true;
  private pointSize = 0.06;

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

    // Point size slider
    const sizeGroup = document.createElement("div");
    sizeGroup.className = "param-group";
    const sizeLabel = document.createElement("label");
    sizeLabel.textContent = "点サイズ";
    const sizeValueSpan = document.createElement("span");
    sizeValueSpan.className = "param-value";
    sizeValueSpan.textContent = this.pointSize.toFixed(2);
    sizeLabel.appendChild(sizeValueSpan);

    const sizeInput = document.createElement("input");
    sizeInput.type = "range";
    sizeInput.min = "0.02";
    sizeInput.max = "0.2";
    sizeInput.step = "0.01";
    sizeInput.value = String(this.pointSize);
    sizeInput.addEventListener("input", () => {
      this.pointSize = parseFloat(sizeInput.value);
      sizeValueSpan.textContent = this.pointSize.toFixed(2);
    });
    sizeInput.addEventListener("change", () => {
      this.updateVisualization();
    });

    sizeGroup.appendChild(sizeLabel);
    sizeGroup.appendChild(sizeInput);
    container.appendChild(sizeGroup);
  }

  private renderInfoPanel(): void {
    if (!this.data) return;
    const container = document.getElementById("info-panel")!;
    const m = this.data.metadata;
    const variance = m.pca_explained_variance;

    container.innerHTML = `
      <h3>情報</h3>
      <div class="info-content">
        <p>MNIST手書き数字の10クラス分類を、Stiefel多様体 St(10, 784) 上の重み行列 W で学習。Riemannian SGD により多様体上で最適化し、PCAで3次元に射影して可視化しています。</p>
        <p><strong>PCA寄与率:</strong></p>
        <p class="math">PC1: ${(variance[0]! * 100).toFixed(1)}%, PC2: ${(variance[1]! * 100).toFixed(1)}%, PC3: ${(variance[2]! * 100).toFixed(1)}%</p>
        <p><strong>可視化要素:</strong></p>
        <p>・<span style="color:#4488ff">青</span>〜<span style="color:#ff4444">赤</span> 点群: St(10,784)上のランダム点 (loss値で色付け)</p>
        <p>・<span style="color:#ffd700">金色</span>曲線: Riemannian SGDの最適化パス</p>
        <p>・<span style="color:#ff1744">赤球</span>: 初期点, <span style="color:#00e676">緑球</span>: 最終点</p>
      </div>
    `;
  }

  private updateVisualization(): void {
    if (!this.data) return;
    removeLossChart();
    const obj = buildVisualization(this.data, {
      showPath: this.showPath,
      pointSize: this.pointSize,
    });
    this.scene.setObject(obj);
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
