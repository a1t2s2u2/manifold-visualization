export interface MnistStiefelData {
  metadata: {
    model: string;
    epochs: number;
    learning_rate: number;
    final_accuracy: number;
    final_loss: number;
    pca_explained_variance: number[];
  };
  landscape: {
    points: [number, number, number][];
    losses: number[];
  };
  optimization_path: {
    points: [number, number, number][];
    losses: number[];
  };
  training_history: {
    epoch_losses: number[];
    epoch_accuracies: number[];
  };
}
