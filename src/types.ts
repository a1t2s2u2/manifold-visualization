export interface LandscapeData {
  points: [number, number, number][];
  losses: number[];
}

export interface ViewData {
  pca_explained_variance: number[];
  landscape: LandscapeData;
  path: LandscapeData;
}

export interface MnistStiefelData {
  metadata: {
    model: string;
    epochs: number;
    learning_rate: number;
    final_accuracy: number;
    final_loss: number;
  };
  local: ViewData;
  global: ViewData;
  training_history: {
    epoch_losses: number[];
    epoch_accuracies: number[];
  };
}

export type LandscapeMode = "local" | "global";
