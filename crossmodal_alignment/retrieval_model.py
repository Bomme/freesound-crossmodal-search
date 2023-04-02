import pytorch_lightning as pl
import pytorch_lightning.utilities.distributed as pld
import torch.distributed as dist
import torch
from torch import optim, nn
from pytorch_metric_learning import losses
from torchmetrics import MetricCollection, RetrievalRecall

from crossmodal_alignment.modules.encoder import (
    MultiLayerPerceptron,
    EmbeddingsAdapterAverage,
)
from crossmodal_alignment.modules.text import TorchTextWordEmbedding


class BiEncoder(pl.LightningModule):
    def __init__(self, audio_encoder, text_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.loss = losses.NTXentLoss()

        metrics = MetricCollection(
            {
                "Recall/Recall@01": RetrievalRecall(k=1),
                "Recall/Recall@05": RetrievalRecall(k=5),
                "Recall/Recall@10": RetrievalRecall(k=10),
            }
        )
        self.train_metrics = metrics.clone(postfix="/Train")
        self.val_metrics = metrics.clone(postfix="/Val")
        self.test_metrics = metrics.clone(postfix="/Test")

        self.save_hyperparameters(ignore=["audio_encoder", "text_encoder"])

        self.val_outputs = []
        self.test_outputs = []

    def _step(
        self,
        audio_labels,
        audio_input,
        audio_comparables,
        text_labels,
        text_input,
        text_comparables,
    ):
        a_encoded = self.audio_encoder(audio_input)
        t_encoded = self.text_encoder(text_input)

        indices_tuple = self._get_indices_tuple(
            audio_labels, text_labels, audio_comparables, text_comparables
        )
        loss = self.loss(a_encoded, indices_tuple=indices_tuple, ref_emb=t_encoded)
        return {
            "loss": loss,
            "audio_outputs": (audio_labels, a_encoded, audio_comparables),
            "text_outputs": (text_labels, t_encoded, text_comparables),
        }

    def _get_indices_tuple(self, labels, ref_labels, comparables, ref_comparables):
        matches = torch.empty(
            (len(labels), len(ref_labels)), dtype=torch.bool, device=self.device
        )
        for i, cmprbl in enumerate(comparables):
            for j, r_cmprbl in enumerate(ref_comparables):
                matches[i, j] = cmprbl == r_cmprbl
        diffs = ~matches
        a1_idx, p_idx = torch.where(matches)
        a2_idx, n_idx = torch.where(diffs)
        return a1_idx, p_idx, a2_idx, n_idx

    def training_step(self, batch, batch_idx):
        output = self._step(*batch)

        self.log("train_loss", output["loss"])
        return output

    def validation_step(self, batch, batch_idx):
        output = self._step(*batch)
        self.log("val_loss", output["loss"])
        self.val_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        output = self._step(*batch)

        self.log("test_loss", output["loss"])
        self.test_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        a_labels, a_embs, a_comparables = self._aggregate_outputs(
            [output["audio_outputs"] for output in self.val_outputs]
        )
        t_labels, t_embs, t_comparables = self._aggregate_outputs(
            [output["text_outputs"] for output in self.val_outputs]
        )

        self.retrieval_metrics(
            t_embs, t_comparables, a_embs, a_comparables, self.val_metrics
        )
        self.val_outputs.clear()

    def on_test_epoch_end(self) -> None:
        a_labels, a_embs, a_comparables = self._aggregate_outputs(
            [output["audio_outputs"] for output in self.test_outputs]
        )
        t_labels, t_embs, t_comparables = self._aggregate_outputs(
            [output["text_outputs"] for output in self.test_outputs]
        )

        self.retrieval_metrics(
            t_embs, t_comparables, a_embs, a_comparables, self.test_metrics
        )
        self.test_outputs.clear()

    @staticmethod
    def _aggregate_outputs(outputs):
        """Aggregate and deduplicate epoch outputs for a single modality

        First, aggregate tensors and objects from distributed devices to get a view of
        the entire data. Second, remove all duplicates that possibly got introduced
        through a distributed sampler or from repeated sampling in dataset. E.g., an
        item in one modality (audio) might be paired with multiple from the other
        (text). We want to remove those duplicates.
        """
        all_embs = []
        all_names = []
        all_comparables = []
        for names, embs, comparables in outputs:
            if pld.distributed_available():
                emb_list = pld.gather_all_tensors(embs)
                names_list = [None for _ in range(dist.get_world_size())]
                comparables_list = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(names_list, names)
                dist.all_gather_object(comparables_list, comparables)
                embs = [emb for l in emb_list for emb in l]
                names = [n for l in names_list for n in l]
                comparables = [n for l in comparables_list for n in l]
            all_embs.extend(embs)
            all_names.extend(names)
            all_comparables.extend(comparables)

        uniq_embs, uniq_names, uniq_comparables = [], [], []
        for emb, name, comparable in zip(
            all_embs, all_names, all_comparables, strict=True
        ):
            if name not in uniq_names:
                uniq_names.append(name)
                uniq_embs.append(emb)
                uniq_comparables.append(comparable)
        uniq_embs = torch.stack(uniq_embs)

        return uniq_names, uniq_embs, uniq_comparables

    def retrieval_metrics(
        self,
        query_embs,
        query_comparables,
        ref_embs,
        ref_comparables,
        metric_collection,
    ):
        results = {}
        similarities = self.loss.distance(query_embs, ref_embs)
        query_indicators = (
            torch.arange(query_embs.shape[0], device=self.device)
            .unsqueeze(1)
            .repeat((1, ref_embs.shape[0]))
        )
        targets = torch.empty(similarities.shape, dtype=torch.bool, device=self.device)

        for i, q in enumerate(query_comparables):
            for j, r in enumerate(ref_comparables):
                targets[i, j] = q == r

        metrics_results = metric_collection(similarities, targets, query_indicators)
        for metric in metrics_results:
            self.log(
                metric, metrics_results[metric], sync_dist=pld.distributed_available()
            )
        return results

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class AudioEmbeddingTorchText(BiEncoder):
    def __init__(
        self,
        audio_emb_dim: int,
        target_dim: int,
        ttext_emb_name: str = "fasttext.en.300d",
    ):
        audio_emb_adapter = MultiLayerPerceptron(audio_emb_dim, target_dim, target_dim)
        text_emb = TorchTextWordEmbedding(ttext_emb_name)

        text_encoder = nn.Sequential(
            text_emb, MultiLayerPerceptron(text_emb.dim, target_dim, target_dim)
        )
        super().__init__(audio_emb_adapter, text_encoder)
