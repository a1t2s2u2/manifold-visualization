import type * as THREE from "three";

export interface ManifoldParam {
  key: string;
  label: string;
  type: "range" | "select" | "checkbox";
  min?: number;
  max?: number;
  step?: number;
  value: number | string | boolean;
  options?: { label: string; value: string }[];
}

export interface ManifoldInfo {
  name: string;
  mathSymbol: string;
  description: string;
  dimension: string;
  properties: string[];
}

export interface ManifoldDefinition {
  id: string;
  info: ManifoldInfo;
  defaultParams: ManifoldParam[];
  generate(params: Record<string, number | string | boolean>): THREE.Object3D;
}
