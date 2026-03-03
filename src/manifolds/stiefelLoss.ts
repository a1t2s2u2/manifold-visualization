import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

/**
 * 論理ゲート学習の損失関数可視化 on Stiefel多様体 V(1,3) = S²
 *
 * 単層パーセプトロン ŷ = sigmoid(α·(w₁x₁ + w₂x₂ + b)) のパラメータ
 * (w₁, w₂, b) を単位ノルム制約下で最適化。
 * S² 上の損失ランドスケープをサーフェスとして描画し、
 * リーマン勾配降下法の最適化パスを可視化する。
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
// Gate data definitions
// ============================================================

type GateType = "AND" | "OR" | "XOR" | "NAND";

interface GateSample {
  x: [number, number];
  y: number;
}

const GATE_DATA: Record<GateType, GateSample[]> = {
  AND: [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 0 },
    { x: [1, 0], y: 0 },
    { x: [1, 1], y: 1 },
  ],
  OR: [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 1 },
  ],
  XOR: [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ],
  NAND: [
    { x: [0, 0], y: 1 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ],
};

// ============================================================
// Forward pass and loss computation
// ============================================================

const ALPHA = 5; // sigmoid steepness

function sigmoid(x: number): number {
  if (x > 500) return 1;
  if (x < -500) return 0;
  return 1 / (1 + Math.exp(-x));
}

/** Compute BCE loss for gate learning. w = [w1, w2, b] on S² */
function computeGateLoss(w: [number, number, number], gate: GateType): number {
  const data = GATE_DATA[gate];
  let loss = 0;
  const eps = 1e-7;
  for (const { x, y } of data) {
    const z = ALPHA * (w[0] * x[0] + w[1] * x[1] + w[2]);
    const yhat = sigmoid(z);
    loss -= y * Math.log(yhat + eps) + (1 - y) * Math.log(1 - yhat + eps);
  }
  return loss / data.length;
}

/** Compute Euclidean gradient of BCE loss w.r.t. w */
function computeGateLossGrad(
  w: [number, number, number],
  gate: GateType
): [number, number, number] {
  const data = GATE_DATA[gate];
  const grad: [number, number, number] = [0, 0, 0];
  for (const { x, y } of data) {
    const z = ALPHA * (w[0] * x[0] + w[1] * x[1] + w[2]);
    const yhat = sigmoid(z);
    const diff = yhat - y; // d(BCE)/d(z) * d(z)/d(w_i) simplified
    grad[0] += diff * ALPHA * x[0];
    grad[1] += diff * ALPHA * x[1];
    grad[2] += diff * ALPHA;
  }
  grad[0] /= data.length;
  grad[1] /= data.length;
  grad[2] /= data.length;
  return grad;
}

// ============================================================
// Riemannian optimization on S²
// ============================================================

/** Project Euclidean gradient to tangent space of S² at w:
 *  rgrad = eucGrad - <w, eucGrad> * w */
function riemannianGradS2(
  w: [number, number, number],
  eucGrad: [number, number, number]
): [number, number, number] {
  const dot = w[0] * eucGrad[0] + w[1] * eucGrad[1] + w[2] * eucGrad[2];
  return [
    eucGrad[0] - dot * w[0],
    eucGrad[1] - dot * w[1],
    eucGrad[2] - dot * w[2],
  ];
}

/** Retract to S²: normalize */
function retractS2(v: [number, number, number]): [number, number, number] {
  const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (norm < 1e-10) return [0, 0, 1];
  return [v[0] / norm, v[1] / norm, v[2] / norm];
}

/** Riemannian gradient descent on S² */
function riemannianGDS2(
  w0: [number, number, number],
  gate: GateType,
  lr: number,
  steps: number
): [number, number, number][] {
  const path: [number, number, number][] = [w0];
  let w: [number, number, number] = [...w0];

  for (let t = 0; t < steps; t++) {
    const eucGrad = computeGateLossGrad(w, gate);
    const rgrad = riemannianGradS2(w, eucGrad);
    const wNew: [number, number, number] = [
      w[0] - lr * rgrad[0],
      w[1] - lr * rgrad[1],
      w[2] - lr * rgrad[2],
    ];
    w = retractS2(wNew);
    path.push([...w]);
  }
  return path;
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
function drawLossChart(pathLosses: number[], lossLabel: string): void {
  removeLossChart();

  if (pathLosses.length < 2) return;

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

  const minV = Math.min(...pathLosses);
  const maxV = Math.max(...pathLosses);
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
    ctx.fillText(val.toFixed(2), pad.left - 4, y + 3);
  }

  // X-axis label
  ctx.textAlign = "center";
  ctx.fillStyle = "#8888aa";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.fillText("step", pad.left + plotW / 2, H - 6);

  // Draw optimization loss curve
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

  // Start / end markers
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
  ctx.fillText(pathLosses[0]!.toFixed(3), sx + 7, sy + 3);
  ctx.fillStyle = "#00e676";
  ctx.textAlign = "right";
  ctx.fillText(pathLosses[steps - 1]!.toFixed(3), ex - 7, ey + 3);

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
  ctx.fillText(`min=${minV.toFixed(3)}`, pad.left + plotW, minLossY - 4);
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
  ctx.fillText(maxLoss.toFixed(2), 32, 28);
  ctx.fillText(((maxLoss + minLoss) / 2).toFixed(2), 32, 20 + (H - 40) / 2);
  ctx.fillText(minLoss.toFixed(2), 32, H - 22);

  ctx.fillStyle = "#8888aa";
  ctx.font = "9px monospace";
  ctx.fillText("Loss", 8, 12);

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    opacity: 0.9,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.5, 2, 1);
  sprite.position.set(-2.5, 0, 0);
  return sprite;
}

// ============================================================
// Loss surface on S² (core visualization)
// ============================================================

function createLossSurface(
  gate: GateType,
  resolution: number,
  deformation: number,
  wireframe: boolean
): { mesh: THREE.Mesh; minLoss: number; maxLoss: number } {
  const geo = new THREE.SphereGeometry(1, resolution, resolution);
  const posAttr = geo.getAttribute("position");
  const vertexCount = posAttr.count;

  // Compute loss at each vertex
  const losses: number[] = [];
  let minLoss = Infinity;
  let maxLoss = -Infinity;

  for (let i = 0; i < vertexCount; i++) {
    const x = posAttr.getX(i);
    const y = posAttr.getY(i);
    const z = posAttr.getZ(i);
    // Normalize to ensure on S² (should already be, but just in case)
    const norm = Math.sqrt(x * x + y * y + z * z);
    const w: [number, number, number] = [x / norm, y / norm, z / norm];
    const loss = computeGateLoss(w, gate);
    losses.push(loss);
    if (loss < minLoss) minLoss = loss;
    if (loss > maxLoss) maxLoss = loss;
  }

  const lossRange = maxLoss - minLoss || 1;

  // Apply vertex colors and deformation
  const colors = new Float32Array(vertexCount * 3);
  const color = new THREE.Color();

  for (let i = 0; i < vertexCount; i++) {
    const t = (losses[i]! - minLoss) / lossRange;

    // Color: blue (low loss) → red (high loss), hue from 0.65 to 0.0
    color.setHSL(0.65 * (1 - t), 0.85, 0.5);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;

    // Deformation: radius = 1 + deformation * t (high loss = mountain, low loss = valley)
    const r = 1 + deformation * t;
    const x = posAttr.getX(i);
    const y = posAttr.getY(i);
    const z = posAttr.getZ(i);
    const norm = Math.sqrt(x * x + y * y + z * z);
    posAttr.setXYZ(i, (x / norm) * r, (y / norm) * r, (z / norm) * r);
  }

  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  posAttr.needsUpdate = true;
  geo.computeVertexNormals();

  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    shininess: 60,
    side: THREE.DoubleSide,
    wireframe,
  });

  const mesh = new THREE.Mesh(geo, mat);
  return { mesh, minLoss, maxLoss };
}

// ============================================================
// Optimization path visualization
// ============================================================

function createOptimizationPath(
  path: [number, number, number][],
  deformation: number,
  gate: GateType,
  minLoss: number,
  maxLoss: number
): THREE.Group {
  const group = new THREE.Group();
  const lossRange = maxLoss - minLoss || 1;
  const offset = 1.005; // z-fighting avoidance

  // Compute deformed positions for path points
  const deformedPositions: THREE.Vector3[] = path.map((w) => {
    const loss = computeGateLoss(w, gate);
    const t = (loss - minLoss) / lossRange;
    const r = (1 + deformation * t) * offset;
    return new THREE.Vector3(w[0] * r, w[1] * r, w[2] * r);
  });

  // Gold line
  const linePositions: number[] = [];
  for (const p of deformedPositions) {
    linePositions.push(p.x, p.y, p.z);
  }
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(linePositions, 3)
  );
  const lineMat = new THREE.LineBasicMaterial({
    color: 0xffd700,
    linewidth: 2,
  });
  group.add(new THREE.Line(lineGeo, lineMat));

  // Start point (red sphere)
  const startGeo = new THREE.SphereGeometry(0.04, 12, 12);
  const startMat = new THREE.MeshPhongMaterial({
    color: 0xff1744,
    emissive: 0xd50000,
    emissiveIntensity: 0.5,
    shininess: 80,
  });
  const startSphere = new THREE.Mesh(startGeo, startMat);
  startSphere.position.copy(deformedPositions[0]!);
  group.add(startSphere);

  // End point (green sphere)
  const endGeo = new THREE.SphereGeometry(0.04, 12, 12);
  const endMat = new THREE.MeshPhongMaterial({
    color: 0x00e676,
    emissive: 0x00c853,
    emissiveIntensity: 0.5,
    shininess: 80,
  });
  const endSphere = new THREE.Mesh(endGeo, endMat);
  endSphere.position.copy(deformedPositions[deformedPositions.length - 1]!);
  group.add(endSphere);

  return group;
}

// ============================================================
// Main generate function
// ============================================================

const GATE_LABELS: Record<GateType, string> = {
  AND: "AND ゲート BCE",
  OR: "OR ゲート BCE",
  XOR: "XOR ゲート BCE (線形分離不可)",
  NAND: "NAND ゲート BCE",
};

function generate(
  params: Record<string, number | string | boolean>
): THREE.Object3D {
  const gate = params["gate"] as GateType;
  const deformation = params["deformation"] as number;
  const showPath = params["showPath"] as boolean;
  const learningRate = params["learningRate"] as number;
  const resolution = params["resolution"] as number;
  const wireframe = params["wireframe"] as boolean;

  const group = new THREE.Group();

  // 1. Loss surface on S²
  const { mesh, minLoss, maxLoss } = createLossSurface(
    gate,
    resolution,
    deformation,
    wireframe
  );
  group.add(mesh);

  // 2. Color bar legend
  group.add(createColorBar(minLoss, maxLoss));

  // 3. Optimization path
  let pathLosses: number[] = [];
  if (showPath) {
    // Random initial point on S²
    const rng = createRng(42);
    const raw: [number, number, number] = [
      seededGaussian(rng),
      seededGaussian(rng),
      seededGaussian(rng),
    ];
    const w0 = retractS2(raw);

    const path = riemannianGDS2(w0, gate, learningRate, 150);
    pathLosses = path.map((w) => computeGateLoss(w, gate));

    const pathGroup = createOptimizationPath(
      path,
      deformation,
      gate,
      minLoss,
      maxLoss
    );
    group.add(pathGroup);
  }

  // 4. Loss chart overlay
  drawLossChart(pathLosses, GATE_LABELS[gate]);

  return group;
}

// ============================================================
// Export manifold definition
// ============================================================

export const stiefelLoss: ManifoldDefinition = {
  id: "stiefelLoss",
  info: {
    name: "ゲート学習 (S²)",
    mathSymbol: "S² = V(1,3)",
    description:
      "論理ゲート (AND/OR/XOR/NAND) の単層パーセプトロン学習を S² = V(1,3) 上で可視化。パラメータ (w₁,w₂,b) の単位ノルム制約により球面上の損失ランドスケープを描画し、リーマン勾配降下法の最適化パスを表示します。XOR は単層では線形分離不可能なため、明確な最小点が存在しません。",
    dimension: "2",
    properties: [
      "モデル: ŷ = σ(5·(w₁x₁ + w₂x₂ + b))",
      "損失: BCE (二値交差エントロピー)",
      "制約: (w₁,w₂,b) ∈ S² — Stiefel多様体 V(1,3)",
      "リーマン勾配 = ユークリッド勾配の接空間射影",
      "XOR: 線形分離不可 → 損失面に最小点なし",
    ],
  },
  defaultParams: [
    {
      key: "gate",
      label: "論理ゲート",
      type: "select",
      value: "AND",
      options: [
        { label: "AND", value: "AND" },
        { label: "OR", value: "OR" },
        { label: "XOR", value: "XOR" },
        { label: "NAND", value: "NAND" },
      ],
    },
    {
      key: "deformation",
      label: "損失地形の変形",
      type: "range",
      min: 0,
      max: 1,
      step: 0.05,
      value: 0.3,
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
      min: 0.1,
      max: 5.0,
      step: 0.1,
      value: 1.0,
    },
    {
      key: "resolution",
      label: "球面解像度",
      type: "range",
      min: 16,
      max: 96,
      step: 4,
      value: 48,
    },
    {
      key: "wireframe",
      label: "ワイヤーフレーム",
      type: "checkbox",
      value: false,
    },
  ],
  generate,
};
