import torch
import pytorch_lightning as pl
import torchtext.vocab.vectors as tt_vectors
from torchtext.vocab.vectors import Vectors
from torch import nn
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer


class TorchTextWordEmbedding(pl.LightningModule):
    def __init__(self, pretrained_alias, **kwargs):
        super().__init__()
        self.tokenizer = get_tokenizer("basic_english")
        self.word_embeddings: Vectors = tt_vectors.pretrained_aliases[pretrained_alias](
            **kwargs
        )
        self.embedding_bag = nn.EmbeddingBag.from_pretrained(
            torch.cat(
                (
                    self.word_embeddings.vectors,
                    self.word_embeddings.vectors.mean(dim=0, keepdim=True),
                ),
            ),
        )

        self.vocab = vocab(self.word_embeddings.stoi)
        # TODO add lowercase transform?

        self.dim = self.word_embeddings.dim
        self.padding_idx = -1
        self.unk_token_idx = len(self.word_embeddings)

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        t_inputs = []
        offsets = []
        for line in text:
            tokens = self.tokenizer(line)
            token_ids = [
                self.word_embeddings.stoi.get(token, self.unk_token_idx)
                for token in tokens
            ]

            offsets.append(len(t_inputs))
            t_inputs.extend(token_ids)
        embs = self.embedding_bag(
            torch.as_tensor(t_inputs, device=self.device),
            torch.as_tensor(offsets, device=self.device),
        )
        return embs
