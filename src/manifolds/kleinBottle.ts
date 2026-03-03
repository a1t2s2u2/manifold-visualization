import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

export const kleinBottle: ManifoldDefinition = {
  id: "klein-bottle",
  info: {
    name: "クライン瓶",
    mathSymbol: "K",
    description:
      "クラインの瓶は向き付け不可能な2次元閉曲面です。3次元空間では自己交差なしには埋め込めないため、ここではイマージョン（はめ込み）を表示しています。",
    dimension: "2",
    properties: [
      "コンパクト",
      "連結",
      "向き付け不可能",
      "3次元に埋め込み不可",
      "オイラー標数: 0",
      "基本群: 非可換",
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

    // Figure-8 Klein bottle immersion
    const geometry = new THREE.BufferGeometry();
    const vertices: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    for (let i = 0; i <= res; i++) {
      const u = (i / res) * 2 * Math.PI;
      for (let j = 0; j <= res; j++) {
        const v = (j / res) * 2 * Math.PI;

        // Figure-8 immersion
        const a = 2;
        const cosU = Math.cos(u);
        const sinU = Math.sin(u);
        const cosV = Math.cos(v);
        const sinV = Math.sin(v);
        const x =
          (a + cosU * Math.cos(v / 2) - sinU * Math.sin(v / 2)) * cosV;
        const y =
          (a + cosU * Math.cos(v / 2) - sinU * Math.sin(v / 2)) * sinV;
        const z = sinU * Math.cos(v / 2) + cosU * Math.sin(v / 2);

        vertices.push(x, y, z);

        // Approximate normal
        const eps = 0.001;
        const u1 = u + eps;
        const v1 = v + eps;

        const xu =
          (a +
            Math.cos(u1) * Math.cos(v / 2) -
            Math.sin(u1) * Math.sin(v / 2)) *
            cosV -
          x;
        const yu =
          (a +
            Math.cos(u1) * Math.cos(v / 2) -
            Math.sin(u1) * Math.sin(v / 2)) *
            sinV -
          y;
        const zu =
          Math.sin(u1) * Math.cos(v / 2) +
          Math.cos(u1) * Math.sin(v / 2) -
          z;

        const xv =
          (a + cosU * Math.cos(v1 / 2) - sinU * Math.sin(v1 / 2)) *
            Math.cos(v1) -
          x;
        const yv =
          (a + cosU * Math.cos(v1 / 2) - sinU * Math.sin(v1 / 2)) *
            Math.sin(v1) -
          y;
        const zv =
          sinU * Math.cos(v1 / 2) + cosU * Math.sin(v1 / 2) - z;

        const nx = yu * zv - zu * yv;
        const ny = zu * xv - xu * zv;
        const nz = xu * yv - yu * xv;
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

    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3)
    );
    geometry.setAttribute(
      "normal",
      new THREE.Float32BufferAttribute(normals, 3)
    );
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);

    const material = new THREE.MeshPhongMaterial({
      color: 0xe17055,
      transparent: true,
      opacity: 0.7,
      wireframe,
      side: THREE.DoubleSide,
      shininess: 60,
    });
    group.add(new THREE.Mesh(geometry, material));

    return group;
  },
};
