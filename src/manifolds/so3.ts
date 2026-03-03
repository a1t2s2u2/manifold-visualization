import * as THREE from "three";
import type { ManifoldDefinition } from "../types";

/**
 * SO(3): Special Orthogonal Group - the group of 3D rotations.
 * Topologically, SO(3) ≅ RP^3 (real projective 3-space).
 *
 * Visualized in 3D using the axis-angle representation:
 * Every rotation maps to a point in the ball of radius π,
 * with antipodal points on the boundary identified.
 */

function sampleGaussian(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// Sample uniformly from SO(3) using the subgroup algorithm
function sampleSO3(): { axis: THREE.Vector3; angle: number; matrix: number[] } {
  // Generate random 3x3 matrix from Gaussians, take QR
  const a = [
    sampleGaussian(), sampleGaussian(), sampleGaussian(),
    sampleGaussian(), sampleGaussian(), sampleGaussian(),
    sampleGaussian(), sampleGaussian(), sampleGaussian(),
  ];

  // Gram-Schmidt to get orthogonal matrix
  // Column 1
  let n1 = Math.sqrt(a[0]! * a[0]! + a[3]! * a[3]! + a[6]! * a[6]!);
  a[0]! /= n1; a[3]! /= n1; a[6]! /= n1;

  // Column 2 - subtract projection
  let dot = a[0]! * a[1]! + a[3]! * a[4]! + a[6]! * a[7]!;
  a[1]! -= dot * a[0]!; a[4]! -= dot * a[3]!; a[7]! -= dot * a[6]!;
  let n2 = Math.sqrt(a[1]! * a[1]! + a[4]! * a[4]! + a[7]! * a[7]!);
  a[1]! /= n2; a[4]! /= n2; a[7]! /= n2;

  // Column 3 = cross product of 1 and 2
  a[2] = a[3]! * a[7]! - a[6]! * a[4]!;
  a[5] = a[6]! * a[1]! - a[0]! * a[7]!;
  a[8] = a[0]! * a[4]! - a[3]! * a[1]!;

  // Ensure det = +1
  const det =
    a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) -
    a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) +
    a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!);
  if (det < 0) {
    a[2] = -a[2]!; a[5] = -a[5]!; a[8] = -a[8]!;
  }

  // Extract axis-angle
  const trace = a[0]! + a[4]! + a[8]!;
  const angle = Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2)));
  let ax = 0, ay = 0, az = 0;

  if (Math.abs(angle) > 1e-6 && Math.abs(angle - Math.PI) > 1e-6) {
    const s = 1 / (2 * Math.sin(angle));
    ax = (a[5]! - a[7]!) * s;
    ay = (a[6]! - a[2]!) * s;
    az = (a[1]! - a[3]!) * s;
  } else if (Math.abs(angle) <= 1e-6) {
    ax = 0; ay = 0; az = 1;
  } else {
    // angle ≈ π
    ax = Math.sqrt(Math.max(0, (a[0]! + 1) / 2));
    ay = Math.sqrt(Math.max(0, (a[4]! + 1) / 2));
    az = Math.sqrt(Math.max(0, (a[8]! + 1) / 2));
    if (a[1]! + a[3]! < 0) ay = -ay;
    if (a[2]! + a[6]! < 0) az = -az;
  }

  const len = Math.sqrt(ax * ax + ay * ay + az * az) || 1;
  return {
    axis: new THREE.Vector3(ax / len, ay / len, az / len),
    angle,
    matrix: a,
  };
}

export const so3: ManifoldDefinition = {
  id: "so3",
  info: {
    name: "回転群",
    mathSymbol: "SO(3)",
    description:
      "SO(3) は3次元空間の回転全体からなるリー群です。軸角度表示を用いて、半径πの球内の点として可視化しています（境界の対蹠点は同一視）。V(3,3)∩{det=1} と同型です。",
    dimension: "3",
    properties: [
      "コンパクトリー群",
      "連結",
      "SO(3) ≅ RP^3",
      "基本群: Z/2Z",
      "リー代数: so(3) ≅ R^3",
      "V(2,3) と微分同相",
    ],
  },
  defaultParams: [
    {
      key: "samples",
      label: "サンプル数",
      type: "range",
      min: 200,
      max: 8000,
      step: 200,
      value: 3000,
    },
    {
      key: "showStructure",
      label: "1パラメータ部分群を表示",
      type: "checkbox",
      value: true,
    },
    {
      key: "colorMode",
      label: "色分け",
      type: "select",
      value: "angle",
      options: [
        { label: "回転角度", value: "angle" },
        { label: "回転軸", value: "axis" },
      ],
    },
  ],
  generate(params) {
    const numSamples = params["samples"] as number;
    const showStructure = params["showStructure"] as boolean;
    const colorMode = params["colorMode"] as string;
    const group = new THREE.Group();

    const positions: number[] = [];
    const colors: number[] = [];
    const color = new THREE.Color();

    for (let i = 0; i < numSamples; i++) {
      const rot = sampleSO3();
      const r = rot.angle / Math.PI;
      const px = rot.axis.x * r * 1.5;
      const py = rot.axis.y * r * 1.5;
      const pz = rot.axis.z * r * 1.5;
      positions.push(px, py, pz);

      if (colorMode === "angle") {
        color.setHSL(rot.angle / Math.PI * 0.8, 0.8, 0.55);
      } else {
        color.setRGB(
          Math.abs(rot.axis.x) * 0.8 + 0.2,
          Math.abs(rot.axis.y) * 0.8 + 0.2,
          Math.abs(rot.axis.z) * 0.8 + 0.2
        );
      }
      colors.push(color.r, color.g, color.b);
    }

    const pointsGeo = new THREE.BufferGeometry();
    pointsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    pointsGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const pointsMat = new THREE.PointsMaterial({
      size: 0.025,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
    });
    group.add(new THREE.Points(pointsGeo, pointsMat));

    // Boundary sphere (radius π mapped to 1.5)
    const sphereGeo = new THREE.SphereGeometry(1.5, 32, 32);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0xdfe6e9,
      wireframe: true,
      transparent: true,
      opacity: 0.06,
    });
    group.add(new THREE.Mesh(sphereGeo, sphereMat));

    // 1-parameter subgroups (great circles in axis-angle space)
    if (showStructure) {
      const axes = [
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(1, 1, 0).normalize(),
        new THREE.Vector3(1, 0, 1).normalize(),
        new THREE.Vector3(0, 1, 1).normalize(),
      ];
      const axisColors = [0xff6b6b, 0x51cf66, 0x339af0, 0xfcc419, 0xcc5de8, 0x20c997];

      axes.forEach((axis, idx) => {
        const linePositions: number[] = [];
        for (let t = 0; t <= 100; t++) {
          const angle = (t / 100) * Math.PI;
          const r = angle / Math.PI * 1.5;
          linePositions.push(axis.x * r, axis.y * r, axis.z * r);
        }
        // Negative direction
        for (let t = 0; t <= 100; t++) {
          const angle = (t / 100) * Math.PI;
          const r = angle / Math.PI * 1.5;
          linePositions.push(-axis.x * r, -axis.y * r, -axis.z * r);
        }
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
        const lineMat = new THREE.LineBasicMaterial({
          color: axisColors[idx]!,
          transparent: true,
          opacity: 0.5,
        });
        group.add(new THREE.Line(lineGeo, lineMat));
      });
    }

    return group;
  },
};
