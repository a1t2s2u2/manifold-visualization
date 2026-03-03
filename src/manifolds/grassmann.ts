import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

/**
 * Grassmann manifold Gr(k, n): the set of k-dimensional subspaces of R^n.
 *
 * Gr(k, n) = V(k, n) / O(k)
 * Each point is an equivalence class of orthonormal k-frames spanning
 * the same subspace.
 *
 * Visualization: sample subspaces, represent each by its projection matrix
 * P = V * V^T (n x n symmetric, idempotent), project to 3D.
 */

function sampleGaussian(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

function qrOrthonormalize(matrix: number[][], k: number, n: number): number[][] {
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

// Compute projection matrix P = V * V^T and extract embedding coordinates
function subspaceToProjection(frame: number[][], k: number, n: number): number[] {
  // P[i][j] = sum_l frame[l][i] * frame[l][j]
  const P: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let val = 0;
      for (let l = 0; l < k; l++) {
        val += frame[l]![i]! * frame[l]![j]!;
      }
      P.push(val);
    }
  }
  return P;
}

// Project the upper-triangular projection matrix to R^3
function projectPTo3D(P: number[]): THREE.Vector3 {
  const dim = P.length;
  // Use different linear combinations for x, y, z
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < dim; i++) {
    x += P[i]! * Math.cos((2 * Math.PI * i) / dim + 0.1);
    y += P[i]! * Math.sin((2 * Math.PI * i) / dim + 0.3);
    z += P[i]! * Math.cos((4 * Math.PI * i) / dim + 0.7);
  }
  return new THREE.Vector3(x, y, z);
}

// Geodesic distance between two subspaces (based on principal angles)
function grassmannDistance(P1: number[], P2: number[]): number {
  let dist = 0;
  for (let i = 0; i < P1.length; i++) {
    const d = P1[i]! - P2[i]!;
    dist += d * d;
  }
  return Math.sqrt(dist);
}

// Special case: Gr(1, 3) = RP^2 (real projective plane)
function generateGr13(numSamples: number): THREE.Group {
  const group = new THREE.Group();

  // RP^2 can be visualized as hemisphere with antipodal boundary identification
  // Boy's surface immersion
  const positions: number[] = [];
  const colors: number[] = [];
  const color = new THREE.Color();

  for (let i = 0; i < numSamples; i++) {
    // Sample a line through origin in R^3
    let x = sampleGaussian();
    let y = sampleGaussian();
    let z = sampleGaussian();
    const len = Math.sqrt(x * x + y * y + z * z);
    x /= len; y /= len; z /= len;

    // Ensure we pick a canonical representative (z >= 0)
    if (z < 0) { x = -x; y = -y; z = -z; }

    // Veronese embedding: (x,y,z) -> (x^2, y^2, z^2, xy√2, xz√2, yz√2)
    // Project to 3D
    const vx = x * x - y * y;
    const vy = x * y * Math.SQRT2;
    const vz = x * z * Math.SQRT2;

    positions.push(vx, vy, vz);
    color.setHSL(0.55 + z * 0.3, 0.8, 0.55);
    colors.push(color.r, color.g, color.b);
  }

  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  pointsGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const pointsMat = new THREE.PointsMaterial({
    size: 0.03,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
  });
  group.add(new THREE.Points(pointsGeo, pointsMat));

  return group;
}

export const grassmann: ManifoldDefinition = {
  id: "grassmann",
  info: {
    name: "Grassmann 多様体",
    mathSymbol: "Gr(k, n)",
    description:
      "Grassmann多様体 Gr(k,n) は n次元空間の k次元部分空間全体の集合です。Stiefel多様体の商空間 V(k,n)/O(k) として得られます。Gr(1,n) は実射影空間 RP^(n-1) に一致します。",
    dimension: "k(n - k)",
    properties: [
      "コンパクト",
      "連結",
      "Gr(1, n) = RP^(n-1)",
      "商空間: V(k,n)/O(k)",
      "対称空間",
      "Plücker埋め込み可能",
    ],
  },
  defaultParams: [
    {
      key: "k",
      label: "部分空間次元 k",
      type: "range",
      min: 1,
      max: 4,
      step: 1,
      value: 1,
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
      min: 200,
      max: 20000,
      step: 100,
      value: 2000,
    },
  ],
  generate(params) {
    let k = params["k"] as number;
    let n = params["n"] as number;
    const samples = params["samples"] as number;

    if (k >= n) k = n - 1;
    if (k < 1) k = 1;

    if (k === 1 && n === 3) return generateGr13(samples);

    const group = new THREE.Group();
    const positions: number[] = [];
    const colors: number[] = [];
    const color = new THREE.Color();
    const projections: number[][] = [];

    for (let i = 0; i < samples; i++) {
      const matrix: number[][] = [];
      for (let j = 0; j < k; j++) {
        const col: number[] = [];
        for (let d = 0; d < n; d++) col.push(sampleGaussian());
        matrix.push(col);
      }
      const frame = qrOrthonormalize(matrix, k, n);
      const P = subspaceToProjection(frame, k, n);
      projections.push(P);
      const p = projectPTo3D(P);
      positions.push(p.x, p.y, p.z);

      const val = (P[0]! + 1) / 2;
      color.setHSL(0.45 + 0.4 * val, 0.75, 0.55);
      colors.push(color.r, color.g, color.b);
    }

    const pointsGeo = new THREE.BufferGeometry();
    pointsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    pointsGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const pointsMat = new THREE.PointsMaterial({
      size: 0.03,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });
    group.add(new THREE.Points(pointsGeo, pointsMat));

    // Connect nearby points
    const linePositions: number[] = [];
    const threshold = 0.3;
    const maxLines = 2000;
    let lineCount = 0;
    for (let i = 0; i < Math.min(projections.length, 200) && lineCount < maxLines; i++) {
      for (let j = i + 1; j < Math.min(projections.length, 200) && lineCount < maxLines; j++) {
        const dist = grassmannDistance(projections[i]!, projections[j]!);
        if (dist < threshold) {
          linePositions.push(
            positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!,
            positions[j * 3]!, positions[j * 3 + 1]!, positions[j * 3 + 2]!
          );
          lineCount++;
        }
      }
    }

    if (linePositions.length > 0) {
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x74b9ff,
        transparent: true,
        opacity: 0.15,
      });
      group.add(new THREE.LineSegments(lineGeo, lineMat));
    }

    return group;
  },
};
