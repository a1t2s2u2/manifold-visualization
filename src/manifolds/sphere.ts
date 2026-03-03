import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

export const sphere: ManifoldDefinition = {
  id: "sphere",
  info: {
    name: "球面",
    mathSymbol: "S^n",
    description:
      "n次元球面はn+1次元ユークリッド空間における単位球面です。最も基本的な多様体の一つで、Stiefel多様体 V(1, n+1) と同型です。",
    dimension: "n",
    properties: [
      "コンパクト",
      "連結",
      "向き付け可能",
      "S^n = V(1, n+1)",
      "オイラー標数: 1+(-1)^n",
    ],
  },
  defaultParams: [
    {
      key: "resolution",
      label: "解像度",
      type: "range",
      min: 8,
      max: 128,
      step: 4,
      value: 48,
    },
    {
      key: "wireframe",
      label: "ワイヤーフレーム表示",
      type: "checkbox",
      value: false,
    },
  ],
  generate(params) {
    const resolution = params["resolution"] as number;
    const wireframe = params["wireframe"] as boolean;
    const group = new THREE.Group();

    const geometry = new THREE.SphereGeometry(1, resolution, resolution);
    const material = new THREE.MeshPhongMaterial({
      color: 0x6c5ce7,
      transparent: true,
      opacity: 0.7,
      wireframe,
      side: THREE.DoubleSide,
      shininess: 80,
    });
    const mesh = new THREE.Mesh(geometry, material);
    group.add(mesh);

    if (!wireframe) {
      const wireGeo = new THREE.SphereGeometry(1.002, 24, 24);
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0xa29bfe,
        wireframe: true,
        transparent: true,
        opacity: 0.1,
      });
      group.add(new THREE.Mesh(wireGeo, wireMat));
    }

    return group;
  },
};
