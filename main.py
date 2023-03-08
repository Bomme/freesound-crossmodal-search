"""Main PytorchLightning CLI interface for model development"""
from pytorch_lightning.cli import LightningCLI

from crossmodal_alignment.retrieval_model import BiEncoder, AudioEmbeddingTorchText
from crossmodal_alignment.datasets import PairedModalitiesDataset
from crossmodal_alignment.datasets.clotho import ClothoEmbeddingsDatamodule

def cli_main():
    cli = LightningCLI(AudioEmbeddingTorchText, ClothoEmbeddingsDatamodule)


if __name__ == "__main__":
    cli_main()
