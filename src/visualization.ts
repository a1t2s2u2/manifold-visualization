import * as THREE from "three";
import type { MnistStiefelData } from "./types";

/**
 * Loss値 → HSL色 (青=低loss, 赤=高loss)
 * hue: 0.65 (blue) → 0.0 (red)
 */
function lossToColor(t: number): THREE.Color {
  return new THREE.Color().setHSL(0.65 * (1 - t), 0.85, 0.5);
}

/**
 * ランドスケープ点群を作成
 */
export function createPointCloud(data: MnistStiefelData): THREE.Points {
  const { points, losses } = data.landscape;
  const n = points.length;

  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const lossRange = maxLoss - minLoss || 1;

  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const [x, y, z] = points[i]!;
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    const t = (losses[i]! - minLoss) / lossRange;
    const color = lossToColor(t);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.06,
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    sizeAttenuation: true,
  });

  return new THREE.Points(geo, mat);
}

/**
 * 最適化パスの曲線 + 始点/終点マーカーを作成
 */
export function createOptimizationPath(data: MnistStiefelData): THREE.Group {
  const { points } = data.optimization_path;
  const group = new THREE.Group();

  if (points.length < 2) return group;

  // Gold line for optimization path
  const linePositions: number[] = [];
  for (const [x, y, z] of points) {
    linePositions.push(x, y, z);
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
  const startGeo = new THREE.SphereGeometry(0.08, 12, 12);
  const startMat = new THREE.MeshPhongMaterial({
    color: 0xff1744,
    emissive: 0xd50000,
    emissiveIntensity: 0.5,
    shininess: 80,
  });
  const startSphere = new THREE.Mesh(startGeo, startMat);
  const [sx, sy, sz] = points[0]!;
  startSphere.position.set(sx, sy, sz);
  group.add(startSphere);

  // End point (green sphere)
  const endGeo = new THREE.SphereGeometry(0.08, 12, 12);
  const endMat = new THREE.MeshPhongMaterial({
    color: 0x00e676,
    emissive: 0x00c853,
    emissiveIntensity: 0.5,
    shininess: 80,
  });
  const endSphere = new THREE.Mesh(endGeo, endMat);
  const last = points[points.length - 1]!;
  endSphere.position.set(last[0], last[1], last[2]);
  group.add(endSphere);

  return group;
}

/**
 * カラーバー凡例 (3Dスプライト)
 */
export function createColorBar(data: MnistStiefelData): THREE.Sprite {
  const losses = data.landscape.losses;
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);

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

/**
 * 全可視化オブジェクトを組み立てる
 */
export function buildVisualization(
  data: MnistStiefelData,
  options: { showPath: boolean; pointSize: number }
): THREE.Group {
  const group = new THREE.Group();

  // Point cloud
  const cloud = createPointCloud(data);
  (cloud.material as THREE.PointsMaterial).size = options.pointSize;
  group.add(cloud);

  // Color bar
  group.add(createColorBar(data));

  // Optimization path
  if (options.showPath) {
    group.add(createOptimizationPath(data));
  }

  return group;
}
