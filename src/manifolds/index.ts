import type { ManifoldDefinition } from "../types";
import { stiefel } from "./stiefel";
import { grassmann } from "./grassmann";
import { sphere } from "./sphere";
import { torus } from "./torus";
import { kleinBottle } from "./kleinBottle";
import { mobiusStrip } from "./mobiusStrip";
import { so3 } from "./so3";
import { realProjective } from "./realProjective";

export const manifolds: ManifoldDefinition[] = [
  stiefel,
  grassmann,
  so3,
  sphere,
  torus,
  realProjective,
  kleinBottle,
  mobiusStrip,
];
