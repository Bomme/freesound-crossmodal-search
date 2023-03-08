from typing import Optional

from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout_prob: Optional[float] = None,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class EmbeddingsAdapterAverage(MultiLayerPerceptron):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout_prob: Optional[float] = None,
    ):
        super().__init__(embedding_dim, output_dim, hidden_dim, dropout_prob)

    def forward(self, x, padding_mask=None):
        adapted_embeddings = self.model(x)
        # TODO: masked mean
        return adapted_embeddings.mean(dim=1)
