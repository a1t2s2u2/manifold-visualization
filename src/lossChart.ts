import type { MnistStiefelData } from "./types";

const CHART_ID = "mnist-loss-chart";

export function removeLossChart(): void {
  document.getElementById(CHART_ID)?.remove();
}

/**
 * 学習ロス収束チャート + エポック精度を2Dオーバーレイで描画
 */
export function drawLossChart(data: MnistStiefelData): void {
  removeLossChart();

  const { epoch_losses, epoch_accuracies } = data.training_history;
  if (epoch_losses.length < 2) return;

  const W = 320;
  const H = 220;
  const pad = { top: 28, right: 16, bottom: 36, left: 52 };

  const canvas = document.createElement("canvas");
  canvas.id = CHART_ID;
  canvas.width = W * 2;
  canvas.height = H * 2;
  canvas.style.cssText = `
    position: fixed; bottom: 16px; left: 16px;
    width: ${W}px; height: ${H}px;
    background: rgba(18, 18, 26, 0.92);
    border: 1px solid rgba(108, 92, 231, 0.4);
    border-radius: 10px;
    pointer-events: none;
    z-index: 10;
  `;
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // --- Loss curve ---
  const minLoss = Math.min(...epoch_losses);
  const maxLoss = Math.max(...epoch_losses);
  const lossRange = maxLoss - minLoss || 1;
  const yMin = minLoss - lossRange * 0.05;
  const yMax = maxLoss + lossRange * 0.05;
  const yRange = yMax - yMin;

  // Title
  ctx.fillStyle = "#a29bfe";
  ctx.font = "bold 11px 'JetBrains Mono', monospace";
  ctx.fillText("Training Loss / Accuracy", pad.left, 16);

  // Grid lines
  ctx.strokeStyle = "rgba(42, 42, 62, 0.8)";
  ctx.lineWidth = 0.5;
  const numGridY = 4;
  for (let i = 0; i <= numGridY; i++) {
    const y = pad.top + (i / numGridY) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
  }

  // Y-axis labels (loss)
  ctx.fillStyle = "#8888aa";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";
  for (let i = 0; i <= numGridY; i++) {
    const y = pad.top + (i / numGridY) * plotH;
    const val = yMax - (i / numGridY) * yRange;
    ctx.fillText(val.toFixed(2), pad.left - 4, y + 3);
  }

  // X-axis label
  ctx.textAlign = "center";
  ctx.fillStyle = "#8888aa";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.fillText("epoch", pad.left + plotW / 2, H - 6);

  // X-axis tick labels
  const epochs = epoch_losses.length;
  for (let i = 0; i < epochs; i++) {
    const x = pad.left + (i / (epochs - 1)) * plotW;
    ctx.fillText(String(i + 1), x, H - 18);
  }

  // Draw loss curve (gold)
  ctx.beginPath();
  ctx.strokeStyle = "#ffd700";
  ctx.lineWidth = 2;
  for (let i = 0; i < epochs; i++) {
    const x = pad.left + (i / (epochs - 1)) * plotW;
    const y = pad.top + ((yMax - epoch_losses[i]!) / yRange) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Loss start/end markers
  const lsx = pad.left;
  const lsy = pad.top + ((yMax - epoch_losses[0]!) / yRange) * plotH;
  ctx.fillStyle = "#ff1744";
  ctx.beginPath();
  ctx.arc(lsx, lsy, 4, 0, Math.PI * 2);
  ctx.fill();

  const lex = pad.left + plotW;
  const ley =
    pad.top + ((yMax - epoch_losses[epochs - 1]!) / yRange) * plotH;
  ctx.fillStyle = "#00e676";
  ctx.beginPath();
  ctx.arc(lex, ley, 4, 0, Math.PI * 2);
  ctx.fill();

  // Draw accuracy curve (cyan, mapped to right axis 0-1)
  ctx.beginPath();
  ctx.strokeStyle = "#00bcd4";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  for (let i = 0; i < epochs; i++) {
    const x = pad.left + (i / (epochs - 1)) * plotW;
    const y = pad.top + (1 - epoch_accuracies[i]!) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Right axis accuracy labels
  ctx.textAlign = "left";
  ctx.fillStyle = "#00bcd4";
  ctx.font = "8px 'JetBrains Mono', monospace";
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * plotH;
    const val = 1 - i / 4;
    ctx.fillText(`${(val * 100).toFixed(0)}%`, pad.left + plotW + 2, y + 3);
  }

  // Legend
  ctx.font = "9px 'JetBrains Mono', monospace";
  const legendY = H - 6;

  ctx.fillStyle = "#ffd700";
  ctx.textAlign = "left";
  ctx.fillText("— Loss", pad.left, legendY);

  ctx.fillStyle = "#00bcd4";
  ctx.fillText("--- Acc", pad.left + 60, legendY);

  // Final values
  const finalLoss = epoch_losses[epochs - 1]!;
  const finalAcc = epoch_accuracies[epochs - 1]!;
  ctx.fillStyle = "#e0e0f0";
  ctx.textAlign = "right";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.fillText(
    `loss=${finalLoss.toFixed(3)}  acc=${(finalAcc * 100).toFixed(1)}%`,
    pad.left + plotW,
    legendY
  );
}
