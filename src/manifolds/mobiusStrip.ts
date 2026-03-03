import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

export const mobiusStrip: ManifoldDefinition = {
  id: "mobius-strip",
  info: {
    name: "メビウスの帯",
    mathSymbol: "M",
    description:
      "メビウスの帯は向き付け不可能な最も単純な曲面です。帯を半回転ひねって端を貼り合わせることで構成されます。境界は一つの閉曲線からなります。",
    dimension: "2 (境界あり)",
    properties: [
      "コンパクト",
      "連結",
      "向き付け不可能",
      "境界は S^1 と同相",
      "オイラー標数: 0",
    ],
  },
  defaultParams: [
    {
      key: "width",
      label: "帯幅",
      type: "range",
      min: 0.2,
      max: 1.5,
      step: 0.1,
      value: 0.5,
    },
    {
      key: "twists",
      label: "ねじれ回数",
      type: "range",
      min: 1,
      max: 5,
      step: 2,
      value: 1,
    },
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
    const width = params["width"] as number;
    const twists = params["twists"] as number;
    const res = params["resolution"] as number;
    const wireframe = params["wireframe"] as boolean;
    const group = new THREE.Group();

    const geometry = new THREE.BufferGeometry();
    const vertices: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];

    const R = 1.0;

    for (let i = 0; i <= res; i++) {
      const u = (i / res) * 2 * Math.PI;
      for (let j = 0; j <= res / 4; j++) {
        const v = (j / (res / 4)) * width - width / 2;

        const halfU = (twists * u) / 2;
        const x = (R + v * Math.cos(halfU)) * Math.cos(u);
        const y = (R + v * Math.cos(halfU)) * Math.sin(u);
        const z = v * Math.sin(halfU);

        vertices.push(x, y, z);

        const nx = Math.cos(halfU) * Math.cos(u);
        const ny = Math.cos(halfU) * Math.sin(u);
        const nz = Math.sin(halfU);
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        normals.push(nx / len, ny / len, nz / len);
      }
    }

    const cols = Math.floor(res / 4) + 1;
    for (let i = 0; i < res; i++) {
      for (let j = 0; j < cols - 1; j++) {
        const a = i * cols + j;
        const b = a + 1;
        const c = (i + 1) * cols + j;
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
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      color: 0xfdcb6e,
      transparent: true,
      opacity: 0.8,
      wireframe,
      side: THREE.DoubleSide,
      shininess: 60,
    });
    group.add(new THREE.Mesh(geometry, material));

    // Add edge highlight
    if (!wireframe) {
      const edgePoints: THREE.Vector3[] = [];
      for (let i = 0; i <= 200; i++) {
        const u = (i / 200) * 2 * Math.PI;
        const halfU = (twists * u) / 2;
        const v = width / 2;
        const x = (R + v * Math.cos(halfU)) * Math.cos(u);
        const y = (R + v * Math.cos(halfU)) * Math.sin(u);
        const z = v * Math.sin(halfU);
        edgePoints.push(new THREE.Vector3(x, y, z));
      }
      const edgeGeo = new THREE.BufferGeometry().setFromPoints(edgePoints);
      const edgeMat = new THREE.LineBasicMaterial({
        color: 0xffeaa7,
        linewidth: 2,
      });
      group.add(new THREE.Line(edgeGeo, edgeMat));
    }

    return group;
  },
};
