import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

/**
 * Stiefel多様体上の損失関数可視化
 *
 * V(k, n) 上の重み行列 W に対して損失関数を評価し、
 * 損失の分布、最小点、リーマン勾配降下法の最適化パスを可視化する。
 */

// ============================================================
// Seeded random number generator (xorshift32)
// ============================================================
function createRng(seed: number): () => number {
  let s = seed | 0;
  if (s === 0) s = 1;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

function seededGaussian(rng: () => number): number {
  const u1 = rng() + 1e-10;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ============================================================
// Matrix utilities (k×n stored as number[][] where mat[row][col])
// ============================================================

/** Multiply A (m×p) by B (p×q), returning m×q */
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const p = A[0]!.length;
  const q = B[0]!.length;
  const C: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = new Array(q).fill(0);
    for (let j = 0; j < q; j++) {
      let s = 0;
      for (let l = 0; l < p; l++) s += A[i]![l]! * B[l]![j]!;
      row[j] = s;
    }
    C.push(row);
  }
  return C;
}

/** Transpose m×n → n×m */
function matTranspose(A: number[][]): number[][] {
  const m = A.length;
  const n = A[0]!.length;
  const T: number[][] = [];
  for (let j = 0; j < n; j++) {
    const row: number[] = [];
    for (let i = 0; i < m; i++) row.push(A[i]![j]!);
    T.push(row);
  }
  return T;
}

/** Trace of a square matrix */
function trace(A: number[][]): number {
  let s = 0;
  for (let i = 0; i < A.length; i++) s += A[i]![i]!;
  return s;
}

/** Symmetrize: (A + A^T) / 2 */
function symmetrize(A: number[][]): number[][] {
  const n = A.length;
  const S: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push((A[i]![j]! + A[j]![i]!) / 2);
    }
    S.push(row);
  }
  return S;
}

/** Frobenius norm squared */
function frobeniusNormSq(A: number[][]): number {
  let s = 0;
  for (const row of A) for (const v of row) s += v * v;
  return s;
}

/** Matrix subtraction A - B */
function matSub(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v - B[i]![j]!));
}

/** Scalar multiply */
function matScale(A: number[][], c: number): number[][] {
  return A.map((row) => row.map((v) => v * c));
}

// ============================================================
// Stiefel manifold operations
// ============================================================

/** QR-based retraction onto V(k, n): columns of W (k×n stored as k rows of length n) */
function qrRetract(W: number[][]): number[][] {
  const k = W.length;
  const n = W[0]!.length;
  const q: number[][] = [];
  for (let j = 0; j < k; j++) {
    const v = W[j]!.slice();
    for (let i = 0; i < j; i++) {
      const qi = q[i]!;
      let dot = 0;
      for (let d = 0; d < n; d++) dot += v[d]! * qi[d]!;
      for (let d = 0; d < n; d++) v[d]! -= dot * qi[d]!;
    }
    let norm = 0;
    for (let d = 0; d < n; d++) norm += v[d]! * v[d]!;
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let d = 0; d < n; d++) v[d]! /= norm;
    }
    q.push(v);
  }
  return q;
}

/** Sample random point on V(k, n) */
function sampleStiefelPoint(k: number, n: number, rng: () => number): number[][] {
  const matrix: number[][] = [];
  for (let j = 0; j < k; j++) {
    const col: number[] = [];
    for (let d = 0; d < n; d++) col.push(seededGaussian(rng));
    matrix.push(col);
  }
  return qrRetract(matrix);
}

/** Project Euclidean gradient to Stiefel tangent space.
 *  Standard formula (column convention): rgrad = G - W sym(W^T G)
 *  W stored as k×n (row convention), so W_row = W_col^T.
 *  Wc^T Gc = W_row · G_row^T (k×k), then rgrad_row[i][d] = G[i][d] - sum_j sym[j][i] * W[j][d] */
function projectToTangent(W: number[][], eucGrad: number[][]): number[][] {
  const k = W.length;
  const n = W[0]!.length;

  // Compute W_row · G_row^T (k×k) = Wc^T · Gc in column convention
  const WGt: number[][] = [];
  for (let a = 0; a < k; a++) {
    const row: number[] = [];
    for (let b = 0; b < k; b++) {
      let s = 0;
      for (let d = 0; d < n; d++) s += W[a]![d]! * eucGrad[b]![d]!;
      row.push(s);
    }
    WGt.push(row);
  }
  const symWGt = symmetrize(WGt);

  // rgrad_row[i][d] = G[i][d] - sum_j sym[j][i] * W[j][d]
  const rgrad: number[][] = [];
  for (let i = 0; i < k; i++) {
    const row: number[] = [];
    for (let d = 0; d < n; d++) {
      let s = eucGrad[i]![d]!;
      for (let j = 0; j < k; j++) {
        s -= symWGt[j]![i]! * W[j]![d]!;
      }
      row.push(s);
    }
    rgrad.push(row);
  }
  return rgrad;
}

// ============================================================
// SPD matrix generation (seeded)
// ============================================================

/** Generate n×n symmetric positive definite matrix using seeded RNG */
function generateSPDMatrix(n: number, seed: number): number[][] {
  const rng = createRng(seed);
  // Generate random matrix and compute A = M^T M + I
  const M: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) row.push(seededGaussian(rng));
    M.push(row);
  }
  const Mt = matTranspose(M);
  const MtM = matMul(Mt, M);
  // Add identity to ensure positive definiteness
  for (let i = 0; i < n; i++) MtM[i]![i]! += 1;
  return MtM;
}

/** Generate n×k target matrix B for linear regression */
function generateTargetMatrix(n: number, k: number, seed: number): number[][] {
  const rng = createRng(seed + 12345);
  const B: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < k; j++) row.push(seededGaussian(rng));
    B.push(row);
  }
  return B;
}

// ============================================================
// Loss functions and their Euclidean gradients
// W is k×n (row convention), A is n×n SPD
// ============================================================

type LossType = "rayleigh" | "brockett" | "regression";

/** Compute loss and Euclidean gradient for the given loss type.
 *  W: k×n, A: n×n, B: n×k (for regression) */
function computeLossAndGrad(
  W: number[][],
  A: number[][],
  B: number[][],
  lossType: LossType,
  k: number,
  _n: number
): { loss: number; grad: number[][] } {
  // W is k×n (row format). In standard math, W would be n×k (column format).
  // Our "W_row" = W_col^T.
  // For the formulas, let's use column convention internally:
  // W_col = W_row^T (n×k)

  const Wt = matTranspose(W); // n×k = W_col

  switch (lossType) {
    case "rayleigh": {
      // L(W) = -tr(W_col^T A W_col) = -tr(W_row A W_row^T)
      const AW = matMul(A, Wt);        // n×k
      const WtAW = matMul(W, AW);      // k×k
      const loss = -trace(WtAW);
      // Euclidean gradient w.r.t. W_col: dL/dW_col = -2AW_col = -2AW (n×k)
      const gradCol = matScale(AW, -2); // n×k
      const grad = matTranspose(gradCol); // k×n (row format)
      return { loss, grad };
    }
    case "brockett": {
      // N = diag(1, 2, ..., k)
      const N: number[][] = [];
      for (let i = 0; i < k; i++) {
        const row: number[] = new Array(k).fill(0);
        row[i] = i + 1;
        N.push(row);
      }
      // L(W) = tr(W_col^T A W_col N)
      const AW = matMul(A, Wt);          // n×k
      const WtAW = matMul(W, AW);        // k×k
      const WtAWN = matMul(WtAW, N);     // k×k
      const loss = trace(WtAWN);
      // Euclidean gradient w.r.t. W_col: 2 A W_col N (n×k)
      const AWN = matMul(AW, N);          // n×k
      const gradCol = matScale(AWN, 2);   // n×k
      const grad = matTranspose(gradCol); // k×n
      return { loss, grad };
    }
    case "regression": {
      // L(W) = ||B - A W_col||_F^2, B is n×k, A is n×n, W_col is n×k
      const AW = matMul(A, Wt);                 // n×k
      const residual = matSub(B, AW);            // n×k
      const loss = frobeniusNormSq(residual);
      // Euclidean gradient w.r.t. W_col: -2 A^T (B - A W_col) = -2 A^T residual
      // A is symmetric so A^T = A
      const AtRes = matMul(matTranspose(A), residual); // n×k
      const gradCol = matScale(AtRes, -2);              // n×k
      const grad = matTranspose(gradCol);                // k×n
      return { loss, grad };
    }
  }
}

// ============================================================
// Riemannian gradient descent on Stiefel manifold
// ============================================================

function riemannianGD(
  W0: number[][],
  A: number[][],
  B: number[][],
  lossType: LossType,
  k: number,
  n: number,
  lr: number,
  steps: number
): number[][][] {
  const path: number[][][] = [W0.map((r) => r.slice())];
  let W = W0.map((r) => r.slice());

  for (let t = 0; t < steps; t++) {
    const { grad } = computeLossAndGrad(W, A, B, lossType, k, n);
    const rgrad = projectToTangent(W, grad);
    // Update: W_new = qr_retract(W - lr * rgrad)
    const Wnew: number[][] = [];
    for (let i = 0; i < k; i++) {
      const row: number[] = [];
      for (let d = 0; d < n; d++) {
        row.push(W[i]![d]! - lr * rgrad[i]![d]!);
      }
      Wnew.push(row);
    }
    W = qrRetract(Wnew);
    path.push(W.map((r) => r.slice()));
  }
  return path;
}

// ============================================================
// 3D projection for visualization
// ============================================================

function projectFrameTo3D(frame: number[][], k: number, n: number): THREE.Vector3 {
  const flat: number[] = [];
  for (let j = 0; j < k; j++) {
    for (let d = 0; d < n; d++) {
      flat.push(frame[j]![d]!);
    }
  }
  const dim = k * n;
  const x = flat.reduce((a, b, i) => a + b * Math.cos((2 * Math.PI * i) / dim), 0);
  const y = flat.reduce((a, b, i) => a + b * Math.sin((2 * Math.PI * i) / dim), 0);
  const z =
    flat.length > 2
      ? flat
          .slice(2, Math.min(dim, 5))
          .reduce((a, b) => a + b, 0) /
        Math.sqrt(Math.min(dim - 2, 3) || 1)
      : 0;
  return new THREE.Vector3(x, y, z);
}

// ============================================================
// Loss chart overlay (2D canvas drawn on DOM)
// ============================================================

const CHART_ID = "stiefel-loss-chart";

/** Remove the loss chart overlay from the DOM */
export function removeLossChart(): void {
  document.getElementById(CHART_ID)?.remove();
}

/** Draw a loss convergence chart as a DOM overlay */
function drawLossChart(
  pathLosses: number[],
  sampleLosses: number[],
  lossLabel: string
): void {
  removeLossChart();

  const W = 320;
  const H = 200;
  const pad = { top: 28, right: 16, bottom: 32, left: 52 };

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

  // Compute ranges
  const allVals = [...pathLosses, ...sampleLosses];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;
  const yMin = minV - range * 0.05;
  const yMax = maxV + range * 0.05;
  const yRange = yMax - yMin;

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Title
  ctx.fillStyle = "#a29bfe";
  ctx.font = "bold 11px 'JetBrains Mono', monospace";
  ctx.fillText(`Loss: ${lossLabel}`, pad.left, 16);

  // Draw sample loss distribution as faint histogram background
  if (sampleLosses.length > 0) {
    const numBins = 30;
    const bins = new Array(numBins).fill(0);
    for (const v of sampleLosses) {
      const bi = Math.min(numBins - 1, Math.floor(((v - yMin) / yRange) * numBins));
      bins[bi]++;
    }
    const maxBin = Math.max(...bins, 1);
    ctx.fillStyle = "rgba(108, 92, 231, 0.12)";
    for (let i = 0; i < numBins; i++) {
      const barH = (bins[i]! / maxBin) * plotW * 0.3;
      const cy = pad.top + plotH - ((i + 0.5) / numBins) * plotH;
      ctx.fillRect(pad.left, cy - plotH / numBins / 2, barH, plotH / numBins);
    }
  }

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

  // Y-axis labels
  ctx.fillStyle = "#8888aa";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";
  for (let i = 0; i <= numGridY; i++) {
    const y = pad.top + (i / numGridY) * plotH;
    const val = yMax - (i / numGridY) * yRange;
    ctx.fillText(val.toFixed(1), pad.left - 4, y + 3);
  }

  // X-axis label
  ctx.textAlign = "center";
  ctx.fillStyle = "#8888aa";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.fillText("step", pad.left + plotW / 2, H - 6);

  // Draw optimization loss curve
  if (pathLosses.length > 1) {
    const steps = pathLosses.length;
    ctx.beginPath();
    ctx.strokeStyle = "#ffd700";
    ctx.lineWidth = 2;
    for (let i = 0; i < steps; i++) {
      const x = pad.left + (i / (steps - 1)) * plotW;
      const y = pad.top + ((yMax - pathLosses[i]!) / yRange) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Start / end markers on chart
    const sx = pad.left;
    const sy = pad.top + ((yMax - pathLosses[0]!) / yRange) * plotH;
    ctx.fillStyle = "#ff1744";
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.fill();

    const ex = pad.left + plotW;
    const ey = pad.top + ((yMax - pathLosses[steps - 1]!) / yRange) * plotH;
    ctx.fillStyle = "#00e676";
    ctx.beginPath();
    ctx.arc(ex, ey, 4, 0, Math.PI * 2);
    ctx.fill();

    // Value labels
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = "#ff1744";
    ctx.fillText(pathLosses[0]!.toFixed(2), sx + 7, sy + 3);
    ctx.fillStyle = "#00e676";
    ctx.textAlign = "right";
    ctx.fillText(pathLosses[steps - 1]!.toFixed(2), ex - 7, ey + 3);
  }

  // Min loss line
  const minLossY = pad.top + ((yMax - minV) / yRange) * plotH;
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = "rgba(0, 230, 118, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, minLossY);
  ctx.lineTo(pad.left + plotW, minLossY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#00e676";
  ctx.textAlign = "right";
  ctx.font = "8px 'JetBrains Mono', monospace";
  ctx.fillText(`min=${minV.toFixed(2)}`, pad.left + plotW, minLossY - 4);
}

// ============================================================
// Color bar legend (3D sprite in scene)
// ============================================================

function createColorBar(minLoss: number, maxLoss: number): THREE.Sprite {
  const W = 64;
  const H = 256;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;

  // Gradient bar
  for (let i = 0; i < H - 40; i++) {
    const t = i / (H - 41);
    const c = new THREE.Color().setHSL(0.65 * (1 - t), 0.85, 0.5);
    ctx.fillStyle = `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
    ctx.fillRect(8, 20 + i, 20, 1);
  }

  // Labels
  ctx.fillStyle = "#e0e0f0";
  ctx.font = "11px monospace";
  ctx.textAlign = "left";
  ctx.fillText(maxLoss.toFixed(1), 32, 28);
  ctx.fillText(((maxLoss + minLoss) / 2).toFixed(1), 32, 20 + (H - 40) / 2);
  ctx.fillText(minLoss.toFixed(1), 32, H - 22);

  ctx.fillStyle = "#8888aa";
  ctx.font = "9px monospace";
  ctx.fillText("Loss", 8, 12);

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.9 });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.5, 2, 1);
  sprite.position.set(-2.5, 0, 0);
  return sprite;
}

// ============================================================
// Main generate function
// ============================================================

const LOSS_LABELS: Record<LossType, string> = {
  rayleigh: "Rayleigh -tr(WᵀAW)",
  brockett: "Brockett tr(WᵀAWN)",
  regression: "Regression ‖B-AW‖²",
};

function generate(params: Record<string, number | string | boolean>): THREE.Object3D {
  let k = params["k"] as number;
  let n = params["n"] as number;
  const numSamples = params["samples"] as number;
  const lossType = params["lossFunction"] as LossType;
  const showPath = params["showPath"] as boolean;
  const lr = params["learningRate"] as number;

  if (k > n) k = n;

  const group = new THREE.Group();
  const seed = 42;

  // Generate SPD matrix A (n×n) and target B (n×k)
  const A = generateSPDMatrix(n, seed);
  const B = generateTargetMatrix(n, k, seed);

  // Sample points on V(k, n) and compute losses
  const rng = createRng(seed + 7);
  const frames: number[][][] = [];
  const losses: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    const frame = sampleStiefelPoint(k, n, rng);
    frames.push(frame);
    const { loss } = computeLossAndGrad(frame, A, B, lossType, k, n);
    losses.push(loss);
  }

  // Find loss range for color mapping
  let minLoss = Infinity;
  let maxLoss = -Infinity;
  let minIdx = 0;
  for (let i = 0; i < losses.length; i++) {
    if (losses[i]! < minLoss) {
      minLoss = losses[i]!;
      minIdx = i;
    }
    if (losses[i]! > maxLoss) maxLoss = losses[i]!;
  }

  const lossRange = maxLoss - minLoss || 1;

  // Point cloud colored by loss value (larger, more visible)
  const positions: number[] = [];
  const colors: number[] = [];
  const sizes: number[] = [];
  const color = new THREE.Color();

  for (let i = 0; i < frames.length; i++) {
    const p = projectFrameTo3D(frames[i]!, k, n);
    positions.push(p.x, p.y, p.z);

    // Blue (low loss) → Red (high loss): hue from 0.65 to 0.0
    const t = (losses[i]! - minLoss) / lossRange;
    color.setHSL(0.65 * (1 - t), 0.9, 0.55);
    colors.push(color.r, color.g, color.b);

    // Low-loss points slightly larger for emphasis
    sizes.push(0.06 - t * 0.02);
  }

  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  pointsGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const pointsMat = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    sizeAttenuation: true,
  });
  group.add(new THREE.Points(pointsGeo, pointsMat));

  // Color bar legend
  group.add(createColorBar(minLoss, maxLoss));

  // Minimum loss marker (green sphere)
  const minPos = projectFrameTo3D(frames[minIdx]!, k, n);
  const minSphereGeo = new THREE.SphereGeometry(0.08, 16, 16);
  const minSphereMat = new THREE.MeshPhongMaterial({
    color: 0x00e676,
    emissive: 0x00c853,
    emissiveIntensity: 0.5,
    shininess: 100,
  });
  const minSphere = new THREE.Mesh(minSphereGeo, minSphereMat);
  minSphere.position.copy(minPos);
  group.add(minSphere);

  // Optimization path via Riemannian gradient descent
  let pathLosses: number[] = [];
  if (showPath) {
    // Start from a random initial point (use a different seed)
    const initRng = createRng(seed + 999);
    const W0 = sampleStiefelPoint(k, n, initRng);
    const path = riemannianGD(W0, A, B, lossType, k, n, lr, 100);

    // Compute loss at each step
    pathLosses = path.map(
      (W) => computeLossAndGrad(W, A, B, lossType, k, n).loss
    );

    // Draw path as gold line
    const linePositions: number[] = [];
    for (const W of path) {
      const p = projectFrameTo3D(W, k, n);
      linePositions.push(p.x, p.y, p.z);
    }
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xffd700,
      linewidth: 2,
    });
    group.add(new THREE.Line(lineGeo, lineMat));

    // Start point marker (red sphere)
    const startPos = projectFrameTo3D(path[0]!, k, n);
    const startGeo = new THREE.SphereGeometry(0.06, 12, 12);
    const startMat = new THREE.MeshPhongMaterial({
      color: 0xff1744,
      emissive: 0xd50000,
      emissiveIntensity: 0.4,
      shininess: 80,
    });
    const startSphere = new THREE.Mesh(startGeo, startMat);
    startSphere.position.copy(startPos);
    group.add(startSphere);

    // End point marker (green sphere)
    const endPos = projectFrameTo3D(path[path.length - 1]!, k, n);
    const endGeo = new THREE.SphereGeometry(0.06, 12, 12);
    const endMat = new THREE.MeshPhongMaterial({
      color: 0x00e676,
      emissive: 0x00c853,
      emissiveIntensity: 0.4,
      shininess: 80,
    });
    const endSphere = new THREE.Mesh(endGeo, endMat);
    endSphere.position.copy(endPos);
    group.add(endSphere);
  }

  // Draw 2D loss chart overlay
  drawLossChart(pathLosses, losses, LOSS_LABELS[lossType]);

  return group;
}

// ============================================================
// Export manifold definition
// ============================================================

export const stiefelLoss: ManifoldDefinition = {
  id: "stiefelLoss",
  info: {
    name: "Stiefel 損失関数",
    mathSymbol: "V(k,n) Loss",
    description:
      "Stiefel多様体 V(k,n) 上の重み行列 W に対して損失関数を評価し、損失分布の可視化とリーマン勾配降下法による最適化パスを表示します。青=低損失、赤=高損失。",
    dimension: "nk - k(k+1)/2",
    properties: [
      "3種の損失関数: Rayleigh商, Brockett, 線形回帰",
      "リーマン勾配 = ユークリッド勾配の接空間射影",
      "QR retraction による多様体上の更新",
      "金色の線: 最適化パス（赤→緑）",
    ],
  },
  defaultParams: [
    {
      key: "k",
      label: "フレーム次元 k",
      type: "range",
      min: 1,
      max: 5,
      step: 1,
      value: 2,
    },
    {
      key: "n",
      label: "空間次元 n",
      type: "range",
      min: 2,
      max: 8,
      step: 1,
      value: 4,
    },
    {
      key: "samples",
      label: "サンプル数",
      type: "range",
      min: 100,
      max: 5000,
      step: 100,
      value: 1000,
    },
    {
      key: "lossFunction",
      label: "損失関数",
      type: "select",
      value: "rayleigh",
      options: [
        { label: "Rayleigh商", value: "rayleigh" },
        { label: "Brockett", value: "brockett" },
        { label: "線形回帰", value: "regression" },
      ],
    },
    {
      key: "showPath",
      label: "最適化パス表示",
      type: "checkbox",
      value: true,
    },
    {
      key: "learningRate",
      label: "学習率",
      type: "range",
      min: 0.001,
      max: 0.1,
      step: 0.001,
      value: 0.01,
    },
  ],
  generate,
};
