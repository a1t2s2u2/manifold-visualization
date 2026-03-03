import type { MnistStiefelData } from "./types";

const DATA_URL = "/data/mnist_stiefel.json";

export async function loadMnistStiefelData(): Promise<MnistStiefelData> {
  const res = await fetch(DATA_URL);
  if (!res.ok) {
    throw new Error(`Failed to load data: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
