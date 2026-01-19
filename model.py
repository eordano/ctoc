"""
Small neural network to predict token counts.

Architecture: Byte-level CNN with regression head.
Target: <1ms inference on CPU, ~98% accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCounterCNN(nn.Module):
    """
    Byte-level CNN for predicting token counts.

    Input: raw bytes (0-255)
    Output: predicted token count (float, round to int)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        max_len: int = 8192,
    ):
        super().__init__()

        # Byte embedding (256 possible values + padding)
        self.embedding = nn.Embedding(257, embed_dim, padding_idx=256)

        # Convolutional layers with increasing dilation
        self.convs = nn.ModuleList()
        in_channels = embed_dim
        for i in range(num_layers):
            dilation = 2 ** i
            self.convs.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation // 2,
                    dilation=dilation,
                )
            )
            in_channels = hidden_dim

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) tensor of byte values (0-255)

        Returns:
            (batch,) tensor of predicted token counts
        """
        # Embed bytes
        h = self.embedding(x)  # (batch, seq_len, embed_dim)
        h = h.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # Conv layers
        for conv in self.convs:
            h = F.relu(conv(h))

        # Global average pooling
        h = h.mean(dim=2)  # (batch, hidden_dim)

        # Predict count
        count = self.head(h).squeeze(-1)  # (batch,)

        return count


class TokenCounterTransformer(nn.Module):
    """
    Tiny transformer for predicting token counts.

    Slightly more accurate than CNN but slower.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 8192,
    ):
        super().__init__()

        self.embedding = nn.Embedding(257, embed_dim, padding_idx=256)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # Embeddings
        h = self.embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = h + self.pos_embedding(positions)

        # Transformer
        h = self.transformer(h)

        # Pool and predict
        h = h.mean(dim=1)
        count = self.head(h).squeeze(-1)

        return count


def text_to_bytes(text: str, max_len: int = 8192) -> torch.Tensor:
    """Convert text to byte tensor."""
    bytes_list = list(text.encode('utf-8'))[:max_len]
    # Pad to max_len with padding token (256)
    bytes_list = bytes_list + [256] * (max_len - len(bytes_list))
    return torch.tensor(bytes_list, dtype=torch.long)


def predict_count(model: nn.Module, text: str) -> int:
    """Predict token count for a string."""
    model.eval()
    with torch.no_grad():
        x = text_to_bytes(text).unsqueeze(0)
        count = model(x).item()
    return round(count)


# Model size estimates
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    cnn = TokenCounterCNN()
    transformer = TokenCounterTransformer()

    print(f"CNN parameters: {count_parameters(cnn):,}")
    print(f"Transformer parameters: {count_parameters(transformer):,}")

    # Test forward pass
    x = text_to_bytes("hello world").unsqueeze(0)
    print(f"CNN output: {cnn(x).item():.2f}")
    print(f"Transformer output: {transformer(x).item():.2f}")
