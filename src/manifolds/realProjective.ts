import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

/**
 * Real Projective Plane RP^2.
 * Visualized via Boy's surface immersion in R^3.
 *
 * Boy's surface is an immersion of RP^2 in R^3 without singular points
 * (unlike the cross-cap which has a singular point).
 */

export const realProjective: ManifoldDefinition = {
  id: "rp2",
  info: {
    name: "実射影平面",
    mathSymbol: "RP^2",
    description:
      "実射影平面は球面上の対蹠点を同一視して得られる多様体です。ここでは Boy の曲面としてはめ込んで表示しています。Grassmann多様体 Gr(1,3) と同型です。",
    dimension: "2",
    properties: [
      "コンパクト",
      "連結",
      "向き付け不可能",
      "RP^2 = S^2 / {±1}",
      "RP^2 = Gr(1, 3)",
      "オイラー標数: 1",
      "基本群: Z/2Z",
    ],
  },
  defaultParams: [
    {
      key: "resolution",
      label: "解像度",
      type: "range",
      min: 16,
      max: 128,
      step: 4,
      value: 64,
    },
    {
      key: "wireframe",
      label: "ワイヤーフレーム表示",
      type: "checkbox",
      value: false,
    },
  ],
  generate(params) {
    const res = params["resolution"] as number;
    const wireframe = params["wireframe"] as boolean;
    const group = new THREE.Group();

    // Boy's surface parametrization (Apéry)
    const geometry = new THREE.BufferGeometry();
    const vertices: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    const uvs: number[] = [];

    for (let i = 0; i <= res; i++) {
      const u = (i / res) * Math.PI;
      for (let j = 0; j <= res; j++) {
        const v = (j / res) * Math.PI;

        const sinU = Math.sin(u);
        const cosU = Math.cos(u);
        const cosV = Math.cos(v);
        const sin2V = Math.sin(2 * v);

        const sqrt2 = Math.SQRT2;

        // Bryant-Kusner parametrization
        const denom =
          2 -
          sqrt2 * sin2V * Math.sin(3 * u);

        const x =
          (sqrt2 * cosV * cosV * Math.cos(2 * u) +
            cosU * sin2V) /
          denom;
        const y =
          (sqrt2 * cosV * cosV * Math.sin(2 * u) -
            sinU * sin2V) /
          denom;
        const z =
          (3 * cosV * cosV) / denom - 1;

        vertices.push(x, y, z);

        // Numerical normal
        const eps = 0.001;
        const u1 = u + eps;
        const v1 = v + eps;

        const denom1 = 2 - sqrt2 * Math.sin(2 * v) * Math.sin(3 * u1);
        const x1u = (sqrt2 * cosV * cosV * Math.cos(2 * u1) + Math.cos(u1) * sin2V) / denom1 - x;
        const y1u = (sqrt2 * cosV * cosV * Math.sin(2 * u1) - Math.sin(u1) * sin2V) / denom1 - y;
        const z1u = (3 * cosV * cosV) / denom1 - 1 - z;

        const denom2 = 2 - sqrt2 * Math.sin(2 * v1) * Math.sin(3 * u);
        const cosV1 = Math.cos(v1);
        const sin2V1 = Math.sin(2 * v1);
        const x1v = (sqrt2 * cosV1 * cosV1 * Math.cos(2 * u) + cosU * sin2V1) / denom2 - x;
        const y1v = (sqrt2 * cosV1 * cosV1 * Math.sin(2 * u) - sinU * sin2V1) / denom2 - y;
        const z1v = (3 * cosV1 * cosV1) / denom2 - 1 - z;

        const nx = y1u * z1v - z1u * y1v;
        const ny = z1u * x1v - x1u * z1v;
        const nz = x1u * y1v - y1u * x1v;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        normals.push(nx / len, ny / len, nz / len);

        uvs.push(i / res, j / res);
      }
    }

    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const a = i * (res + 1) + j;
        const b = a + 1;
        const c = (i + 1) * (res + 1) + j;
        const d = c + 1;
        indices.push(a, b, c);
        indices.push(b, d, c);
      }
    }

    geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);

    const material = new THREE.MeshPhongMaterial({
      color: 0x0984e3,
      transparent: true,
      opacity: 0.65,
      wireframe,
      side: THREE.DoubleSide,
      shininess: 60,
    });
    group.add(new THREE.Mesh(geometry, material));

    return group;
  },
};
