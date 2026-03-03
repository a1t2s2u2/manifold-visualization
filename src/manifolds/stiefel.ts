import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

/**
 * Stiefel manifold V(k, n): the set of orthonormal k-frames in R^n.
 *
 * For visualization, we sample points on the manifold and project them
 * to 3D. Special cases:
 *   V(1, 2) = S^1 (circle)
 *   V(1, 3) = S^2 (sphere)
 *   V(2, 3) = SO(3) (rotation group, visualized via axis-angle)
 *   V(k, n) for general cases: sample orthonormal frames, project to 3D
 */

// QR decomposition to project onto Stiefel manifold
function qrOrthonormalize(
  matrix: number[][],
  k: number,
  n: number
): number[][] {
  const q: number[][] = [];
  for (let j = 0; j < k; j++) {
    const v = matrix[j]!.slice();
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

// Sample a random point on V(k, n) via QR of Gaussian matrix
function sampleStiefelPoint(k: number, n: number): number[][] {
  const matrix: number[][] = [];
  for (let j = 0; j < k; j++) {
    const col: number[] = [];
    for (let d = 0; d < n; d++) {
      // Box-Muller for standard normal
      const u1 = Math.random();
      const u2 = Math.random();
      col.push(Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2));
    }
    matrix.push(col);
  }
  return qrOrthonormalize(matrix, k, n);
}

// Project a Stiefel point to R^3 for visualization
function projectTo3D(frame: number[][], k: number, n: number): THREE.Vector3 {
  // Use first 3 components of flattened frame, with PCA-like projection
  const flat: number[] = [];
  for (let j = 0; j < k; j++) {
    for (let d = 0; d < n; d++) {
      flat.push(frame[j]![d]!);
    }
  }
  // Take 3 linearly independent projections
  const dim = k * n;
  const x = flat.slice(0, Math.min(dim, 3)).reduce((a, b, i) => a + b * Math.cos((2 * Math.PI * i) / dim), 0);
  const y = flat.slice(0, Math.min(dim, 3)).reduce((a, b, i) => a + b * Math.sin((2 * Math.PI * i) / dim), 0);
  const z = flat.length > 2 ? flat.slice(2, Math.min(dim, 5)).reduce((a, b) => a + b, 0) / Math.sqrt(Math.min(dim - 2, 3) || 1) : 0;
  return new THREE.Vector3(x, y, z);
}

// Generate geodesic path on Stiefel manifold between two points
function stiefelGeodesicPath(
  p1: number[][],
  p2: number[][],
  k: number,
  n: number,
  steps: number
): number[][][] {
  const path: number[][][] = [];
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    // Linear interpolation + retraction onto Stiefel
    const interpolated: number[][] = [];
    for (let j = 0; j < k; j++) {
      const col: number[] = [];
      for (let d = 0; d < n; d++) {
        col.push((1 - t) * p1[j]![d]! + t * p2[j]![d]!);
      }
      interpolated.push(col);
    }
    path.push(qrOrthonormalize(interpolated, k, n));
  }
  return path;
}

// V(1,3) = S^2: parametric surface
function generateV13(resolution: number, wireframe: boolean): THREE.Group {
  const group = new THREE.Group();
  const geometry = new THREE.SphereGeometry(1, resolution, resolution);
  const material = new THREE.MeshPhongMaterial({
    color: 0xe84393,
    transparent: true,
    opacity: 0.6,
    wireframe,
    side: THREE.DoubleSide,
    shininess: 80,
  });
  group.add(new THREE.Mesh(geometry, material));

  // Sample points on S^2 = V(1,3)
  const pointsGeo = new THREE.BufferGeometry();
  const positions: number[] = [];
  const colors: number[] = [];
  const color = new THREE.Color();
  for (let i = 0; i < 200; i++) {
    const frame = sampleStiefelPoint(1, 3);
    const p = frame[0]!;
    positions.push(p[0]!, p[1]!, p[2]!);
    color.setHSL(0.85 + p[2]! * 0.15, 0.9, 0.6);
    colors.push(color.r, color.g, color.b);
  }
  pointsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  pointsGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const pointsMat = new THREE.PointsMaterial({
    size: 0.04,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
  });
  group.add(new THREE.Points(pointsGeo, pointsMat));

  return group;
}

// V(2,3) ≅ SO(3): visualize as rotation group using axis-angle
function generateV23(
  numSamples: number,
  showGeodesics: boolean
): THREE.Group {
  const group = new THREE.Group();

  const positions: number[] = [];
  const colors: number[] = [];
  const color = new THREE.Color();

  const frames: number[][][] = [];

  for (let i = 0; i < numSamples; i++) {
    const frame = sampleStiefelPoint(2, 3);
    frames.push(frame);

    // Construct rotation matrix from the orthonormal 2-frame
    // v1, v2 -> complete to SO(3) via cross product
    const v1 = frame[0]!;
    const v2 = frame[1]!;
    const v3 = [
      v1[1]! * v2[2]! - v1[2]! * v2[1]!,
      v1[2]! * v2[0]! - v1[0]! * v2[2]!,
      v1[0]! * v2[1]! - v1[1]! * v2[0]!,
    ];

    // Extract axis-angle from rotation matrix [v1, v2, v3]
    // trace = v1[0] + v2[1] + v3[2]
    const trace = v1[0]! + v2[1]! + v3[2]!;
    const angle = Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2)));

    let ax = 0,
      ay = 0,
      az = 0;
    if (Math.abs(angle) > 1e-6 && Math.abs(angle - Math.PI) > 1e-6) {
      const s = 1 / (2 * Math.sin(angle));
      ax = (v2[2]! - v3[1]!) * s;
      ay = (v3[0]! - v1[2]!) * s;
      az = (v1[1]! - v2[0]!) * s;
    } else {
      ax = Math.random() - 0.5;
      ay = Math.random() - 0.5;
      az = Math.random() - 0.5;
      const len = Math.sqrt(ax * ax + ay * ay + az * az) || 1;
      ax /= len;
      ay /= len;
      az /= len;
    }

    // Map to ball: direction = axis, radius = angle / PI
    const r = angle / Math.PI;
    const px = ax * r * 1.5;
    const py = ay * r * 1.5;
    const pz = az * r * 1.5;
    positions.push(px, py, pz);

    color.setHSL(angle / Math.PI, 0.8, 0.55);
    colors.push(color.r, color.g, color.b);
  }

  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  pointsGeo.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(colors, 3)
  );
  const pointsMat = new THREE.PointsMaterial({
    size: 0.03,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
  });
  group.add(new THREE.Points(pointsGeo, pointsMat));

  // Bounding sphere for reference
  const sphereGeo = new THREE.SphereGeometry(1.5, 24, 24);
  const sphereMat = new THREE.MeshBasicMaterial({
    color: 0xe84393,
    wireframe: true,
    transparent: true,
    opacity: 0.08,
  });
  group.add(new THREE.Mesh(sphereGeo, sphereMat));

  // Draw some geodesics
  if (showGeodesics && frames.length >= 2) {
    for (let g = 0; g < Math.min(8, frames.length - 1); g++) {
      const path = stiefelGeodesicPath(frames[g]!, frames[g + 1]!, 2, 3, 30);
      const linePositions: number[] = [];
      for (const pt of path) {
        const v1 = pt[0]!;
        const v2 = pt[1]!;
        const v3 = [
          v1[1]! * v2[2]! - v1[2]! * v2[1]!,
          v1[2]! * v2[0]! - v1[0]! * v2[2]!,
          v1[0]! * v2[1]! - v1[1]! * v2[0]!,
        ];
        const trace = v1[0]! + v2[1]! + v3[2]!;
        const angle = Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2)));
        let ax = 0,
          ay = 0,
          az = 0;
        if (Math.abs(angle) > 1e-6 && Math.abs(angle - Math.PI) > 1e-6) {
          const s = 1 / (2 * Math.sin(angle));
          ax = (v2[2]! - v3[1]!) * s;
          ay = (v3[0]! - v1[2]!) * s;
          az = (v1[1]! - v2[0]!) * s;
        }
        const r = angle / Math.PI;
        linePositions.push(ax * r * 1.5, ay * r * 1.5, az * r * 1.5);
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(linePositions, 3)
      );
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xfd79a8,
        transparent: true,
        opacity: 0.4,
      });
      group.add(new THREE.Line(lineGeo, lineMat));
    }
  }

  return group;
}

// General V(k, n): point cloud with geodesics
function generateGeneral(
  k: number,
  n: number,
  numSamples: number,
  showGeodesics: boolean
): THREE.Group {
  const group = new THREE.Group();
  const positions: number[] = [];
  const colors: number[] = [];
  const color = new THREE.Color();
  const frames: number[][][] = [];

  for (let i = 0; i < numSamples; i++) {
    const frame = sampleStiefelPoint(k, n);
    frames.push(frame);
    const p = projectTo3D(frame, k, n);
    positions.push(p.x, p.y, p.z);

    // Color based on first element of frame
    const val = (frame[0]![0]! + 1) / 2;
    color.setHSL(0.8 * val + 0.1, 0.8, 0.55);
    colors.push(color.r, color.g, color.b);
  }

  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  pointsGeo.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(colors, 3)
  );
  const pointsMat = new THREE.PointsMaterial({
    size: 0.035,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
  });
  group.add(new THREE.Points(pointsGeo, pointsMat));

  if (showGeodesics && frames.length >= 2) {
    for (let g = 0; g < Math.min(12, frames.length - 1); g++) {
      const path = stiefelGeodesicPath(
        frames[g]!,
        frames[g + 1]!,
        k,
        n,
        20
      );
      const linePositions: number[] = [];
      for (const pt of path) {
        const p = projectTo3D(pt, k, n);
        linePositions.push(p.x, p.y, p.z);
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(linePositions, 3)
      );
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xfd79a8,
        transparent: true,
        opacity: 0.3,
      });
      group.add(new THREE.Line(lineGeo, lineMat));
    }
  }

  return group;
}

export const stiefel: ManifoldDefinition = {
  id: "stiefel",
  info: {
    name: "Stiefel 多様体",
    mathSymbol: "V(k, n)",
    description:
      "Stiefel多様体 V(k,n) は n次元ユークリッド空間における全ての正規直交 k-フレームの集合です。V(1,n)=S^(n-1)（球面）、V(n,n)=O(n)（直交群）という重要な特殊ケースを含みます。",
    dimension: "nk - k(k+1)/2",
    properties: [
      "コンパクト",
      "連結（k < n のとき）",
      "V(1, n) = S^(n-1)",
      "V(n, n) = O(n)",
      "V(2, 3) ≅ SO(3)",
      "同次空間: O(n)/O(n-k)",
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
      value: 3,
    },
    {
      key: "samples",
      label: "サンプル数",
      type: "range",
      min: 100,
      max: 20000,
      step: 100,
      value: 1500,
    },
    {
      key: "geodesics",
      label: "測地線を表示",
      type: "checkbox",
      value: true,
    },
    {
      key: "wireframe",
      label: "ワイヤーフレーム表示",
      type: "checkbox",
      value: false,
    },
  ],
  generate(params) {
    let k = params["k"] as number;
    let n = params["n"] as number;
    const samples = params["samples"] as number;
    const geodesics = params["geodesics"] as boolean;
    const wireframe = params["wireframe"] as boolean;

    // Ensure k <= n
    if (k > n) k = n;

    // Special cases
    if (k === 1 && n === 3) return generateV13(48, wireframe);
    if (k === 2 && n === 3) return generateV23(samples, geodesics);

    return generateGeneral(k, n, samples, geodesics);
  },
};
