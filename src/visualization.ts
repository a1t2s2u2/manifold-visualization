import * as THREE from "three";
import type { ViewData, LandscapeMode, MnistStiefelData } from "./types";

/**
 * Loss値 → HSL色 (青=低loss, 赤=高loss)
 * hue: 0.65 (blue) → 0.0 (red)
 */
function lossToColor(t: number): THREE.Color {
  return new THREE.Color().setHSL(0.65 * (1 - t), 0.85, 0.5);
}

function lossToCSS(t: number): string {
  const c = new THREE.Color().setHSL(0.65 * (1 - t), 0.85, 0.5);
  return `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
}

/**
 * 点群を作成
 */
function createPointCloud(view: ViewData): THREE.Points {
  const { points, losses } = view.landscape;
  const n = points.length;

  // Use percentile-based normalization for better contrast
  const sorted = [...losses].sort((a, b) => a - b);
  const p2 = sorted[Math.floor(n * 0.02)] ?? sorted[0]!;
  const p98 = sorted[Math.floor(n * 0.98)] ?? sorted[n - 1]!;
  const lossRange = p98 - p2 || 1;

  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const [x, y, z] = points[i]!;
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    const t = Math.max(0, Math.min(1, (losses[i]! - p2) / lossRange));
    const color = lossToColor(t);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.12,
    vertexColors: true,
    transparent: true,
    opacity: 0.75,
    sizeAttenuation: true,
  });

  return new THREE.Points(geo, mat);
}

/**
 * 最適化パスの曲線 + 始点/終点マーカーを作成
 */
function createOptimizationPath(view: ViewData): THREE.Group {
  const { points } = view.path;
  const lPoints = view.landscape.points;
  const group = new THREE.Group();

  if (points.length < 2) return group;

  // Compute landscape spread and path spread
  let lMinX = Infinity, lMaxX = -Infinity;
  let lMinY = Infinity, lMaxY = -Infinity;
  let lMinZ = Infinity, lMaxZ = -Infinity;
  for (const [x, y, z] of lPoints) {
    if (x < lMinX) lMinX = x; if (x > lMaxX) lMaxX = x;
    if (y < lMinY) lMinY = y; if (y > lMaxY) lMaxY = y;
    if (z < lMinZ) lMinZ = z; if (z > lMaxZ) lMaxZ = z;
  }
  const landscapeSpread = Math.max(lMaxX - lMinX, lMaxY - lMinY, lMaxZ - lMinZ) || 1;

  let pMinX = Infinity, pMaxX = -Infinity;
  let pMinY = Infinity, pMaxY = -Infinity;
  let pMinZ = Infinity, pMaxZ = -Infinity;
  for (const [x, y, z] of points) {
    if (x < pMinX) pMinX = x; if (x > pMaxX) pMaxX = x;
    if (y < pMinY) pMinY = y; if (y > pMaxY) pMaxY = y;
    if (z < pMinZ) pMinZ = z; if (z > pMaxZ) pMaxZ = z;
  }
  const pathSpread = Math.max(pMaxX - pMinX, pMaxY - pMinY, pMaxZ - pMinZ);

  // If path spread is tiny compared to landscape (global view),
  // show a single "optimized W" marker instead of start/end
  const isCompact = pathSpread < landscapeSpread * 0.05;
  const markerRadius = landscapeSpread * 0.04;

  if (isCompact) {
    // Global view: prominent marker at final W position (white+magenta to contrast loss colors)
    const last = points[points.length - 1]!;

    // Inner bright white core
    const coreGeo = new THREE.SphereGeometry(markerRadius, 16, 16);
    const coreMat = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      emissive: 0xff44ff,
      emissiveIntensity: 1.0,
      shininess: 120,
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    core.position.set(last[0], last[1], last[2]);
    group.add(core);

    // Outer glow ring
    const glowGeo = new THREE.RingGeometry(markerRadius * 1.8, markerRadius * 2.5, 32);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0xff44ff,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.position.set(last[0], last[1], last[2]);
    group.add(glow);

    // Second ring perpendicular
    const glow2 = new THREE.Mesh(glowGeo.clone(), glowMat.clone());
    glow2.position.set(last[0], last[1], last[2]);
    glow2.rotation.x = Math.PI / 2;
    group.add(glow2);
  } else {
    // Local view: full path with start/end markers
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
    const startGeo = new THREE.SphereGeometry(markerRadius, 12, 12);
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
    const endGeo = new THREE.SphereGeometry(markerRadius, 12, 12);
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
  }

  return group;
}

// ============================================================
// Color bar as DOM overlay (does not affect 3D auto-scaling)
// ============================================================

const COLORBAR_ID = "colorbar-overlay";

export function removeColorBar(): void {
  document.getElementById(COLORBAR_ID)?.remove();
}

function drawColorBar(view: ViewData): void {
  removeColorBar();

  const { losses } = view.landscape;
  const n = losses.length;
  const sorted = [...losses].sort((a, b) => a - b);
  const minLoss = sorted[Math.floor(n * 0.02)] ?? sorted[0]!;
  const maxLoss = sorted[Math.floor(n * 0.98)] ?? sorted[n - 1]!;

  const container = document.createElement("div");
  container.id = COLORBAR_ID;
  container.style.cssText = `
    position: fixed; top: 50%; right: 340px;
    transform: translateY(-50%);
    display: flex; align-items: center; gap: 6px;
    background: rgba(18, 18, 26, 0.85);
    border: 1px solid rgba(108, 92, 231, 0.4);
    border-radius: 8px;
    padding: 10px 12px;
    pointer-events: none;
    z-index: 10;
    font-family: 'JetBrains Mono', monospace;
  `;

  // Gradient strip
  const strip = document.createElement("div");
  const gradH = 160;
  strip.style.cssText = `
    width: 14px; height: ${gradH}px; border-radius: 3px;
    background: linear-gradient(to bottom, ${lossToCSS(1)}, ${lossToCSS(0.5)}, ${lossToCSS(0)});
  `;

  // Labels
  const labels = document.createElement("div");
  labels.style.cssText = `
    display: flex; flex-direction: column;
    justify-content: space-between;
    height: ${gradH}px; font-size: 10px; color: #e0e0f0;
  `;
  labels.innerHTML = `
    <span>${maxLoss.toFixed(2)}</span>
    <span style="color:#8888aa;font-size:9px">Loss</span>
    <span>${minLoss.toFixed(2)}</span>
  `;

  container.appendChild(strip);
  container.appendChild(labels);
  document.body.appendChild(container);
}

/**
 * 全可視化オブジェクトを組み立てる
 */
export function buildVisualization(
  data: MnistStiefelData,
  options: { showPath: boolean; landscapeMode: LandscapeMode }
): THREE.Group {
  const group = new THREE.Group();
  const view = data[options.landscapeMode];

  group.add(createPointCloud(view));

  // Color bar as DOM overlay (no 3D bounding box impact)
  drawColorBar(view);

  if (options.showPath) {
    group.add(createOptimizationPath(view));
  }

  return group;
}
