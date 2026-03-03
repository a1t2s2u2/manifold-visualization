import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

export const torus: ManifoldDefinition = {
  id: "torus",
  info: {
    name: "トーラス",
    mathSymbol: "T^2 = S^1 x S^1",
    description:
      "トーラスは2つの円の直積として得られる2次元多様体です。平坦な計量を持ち、種数1の閉曲面です。",
    dimension: "2",
    properties: [
      "コンパクト",
      "連結",
      "向き付け可能",
      "平坦計量が存在",
      "基本群: Z x Z",
      "オイラー標数: 0",
    ],
  },
  defaultParams: [
    {
      key: "majorRadius",
      label: "主半径 R",
      type: "range",
      min: 0.5,
      max: 2.0,
      step: 0.1,
      value: 1.0,
    },
    {
      key: "minorRadius",
      label: "管半径 r",
      type: "range",
      min: 0.1,
      max: 0.8,
      step: 0.05,
      value: 0.4,
    },
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
    const R = params["majorRadius"] as number;
    const r = params["minorRadius"] as number;
    const res = params["resolution"] as number;
    const wireframe = params["wireframe"] as boolean;
    const group = new THREE.Group();

    const geometry = new THREE.TorusGeometry(R, r, res, res);
    const material = new THREE.MeshPhongMaterial({
      color: 0x00b894,
      transparent: true,
      opacity: 0.75,
      wireframe,
      side: THREE.DoubleSide,
      shininess: 80,
    });
    group.add(new THREE.Mesh(geometry, material));

    if (!wireframe) {
      const wireGeo = new THREE.TorusGeometry(R, r + 0.002, 20, 20);
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x55efc4,
        wireframe: true,
        transparent: true,
        opacity: 0.1,
      });
      group.add(new THREE.Mesh(wireGeo, wireMat));
    }

    return group;
  },
};
