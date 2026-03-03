import type { ManifoldDefinition } from "../types";
import { stiefel } from "./stiefel";
import { stiefelLoss } from "./stiefelLoss";
import { grassmann } from "./grassmann";
import { so3 } from "./so3";
import { sphere } from "./sphere";
import { realProjective } from "./realProjective";

export const manifolds: ManifoldDefinition[] = [
  stiefel,
  stiefelLoss,
  grassmann,
  so3,
  sphere,
  realProjective,
];
