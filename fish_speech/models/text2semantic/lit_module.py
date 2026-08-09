from typing import Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import fish_speech.utils as utils

CODEBOOK_PAD_TOKEN_ID = 0
from fish_speech.models.text2semantic.llama import NaiveTransformer

log = utils.RankedLogger(__name__, rank_zero_only=True)


class TextToSemantic(L.LightningModule):
    def __init__(
        self,
        model: NaiveTransformer,
        optimizer: Any,
        lr_scheduler: Any,
        semantic_loss_weight: float = 1.0,
        base_kl_weight: float = 0.0,
    ):
        super().__init__()

        self.model = model
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        # Weight on the fast-transformer (residual codebook) loss. Qwen3-TTS
        # uses 0.3 for the equivalent sub-talker term; 1.0 matches upstream
        # fish-speech behaviour.
        self.semantic_loss_weight = semantic_loss_weight
        # Anchor to the frozen base model: reverse KL(p_lora || p_base) on the
        # token and codebook distributions at loss positions. Plain CE is
        # teacher-forced and never measures free-running coherence — LoRA
        # fine-tuning S2-Pro on a new language collapses generation (near-
        # silent output) at ANY adapter magnitude while CE happily improves.
        # The anchor penalizes leaving the base model's generative manifold.
        # The base distribution comes from the SAME model with every LoRA
        # module's `scaling` set to 0 (bit-exact base, no extra weights).
        self.base_kl_weight = base_kl_weight

    def forward(self, x):
        return self.model(x)

    def _set_lora_scaling(self, factor: float):
        import loralib

        for module in self.model.modules():
            if isinstance(module, (loralib.Linear, loralib.Embedding)):
                module.scaling = (module.lora_alpha / module.r) * factor

    @staticmethod
    def _reverse_kl(logits, base_logits, mask, chunk: int = 512):
        """KL(p || p_base) averaged over masked positions. Mode-seeking: keeps
        the fine-tuned distribution inside the base model's support.

        Computed over gathered mask positions in chunks — a full fp32 softmax
        over (B, T, 156k vocab) would need ~15 GB."""
        flat_mask = mask.reshape(-1).bool()
        idx = flat_mask.nonzero().squeeze(1)
        if idx.numel() == 0:
            return logits.sum() * 0.0

        p = logits.reshape(-1, logits.size(-1))[idx]
        q = base_logits.reshape(-1, base_logits.size(-1))[idx]

        total = p.new_zeros(())
        for i in range(0, p.size(0), chunk):
            lp = F.log_softmax(p[i : i + chunk].float(), dim=-1)
            lq = F.log_softmax(q[i : i + chunk].float(), dim=-1)
            total = total + (lp.exp() * (lp - lq)).sum()
        return total / idx.numel()

    def on_save_checkpoint(self, checkpoint):
        # Save only LoRA parameters
        state_dict = checkpoint["state_dict"]
        use_lora = any("lora" in name for name in state_dict.keys())
        if not use_lora:
            return

        for name in list(state_dict.keys()):
            if "lora" not in name:
                state_dict.pop(name)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Get weight decay parameters
        weight_decay_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                other_parameters.append(param)
            else:
                weight_decay_parameters.append(param)

        optimizer = self.optimizer_builder(
            [
                {"params": weight_decay_parameters},
                {"params": other_parameters, "weight_decay": 0.0},
            ]
        )

        # Print the parameters and their weight decay
        for i in optimizer.param_groups:
            log.info(
                f"Set weight decay: {i['weight_decay']} for {len(i['params'])} parameters"
            )

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    # Copied from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90
    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, codebook_size, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length, codebook_size)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels.clone()
        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _step(self, batch, batch_idx, stage: str):
        is_train = stage == "train"

        if is_train:
            # Key part to make lora work
            # Otherwise the parameters are merged, which lead to incorrect gradients
            self.model.train()

        # Do positive and negative samples in the same batch to speed up training
        labels = batch["labels"]
        outputs = self.model(
            inp=batch["inputs"],
            key_padding_mask=batch["attention_masks"],
            labels=batch["labels"],
        )
        token_logits = outputs.token_logits
        codebook_logits = outputs.codebook_logits

        # Generate labels
        base_loss = F.cross_entropy(
            token_logits.view(-1, token_logits.size(-1)),
            labels[:, 0].reshape(-1),
            ignore_index=-100,
        )

        token_ids = labels[:, 0]
        semantic_mask = (token_ids >= self.model.tokenizer.semantic_begin_id) & (
            token_ids <= self.model.tokenizer.semantic_end_id
        )
        all_codebook_labels = labels[:, 1 : 1 + self.model.config.num_codebooks]
        all_codebook_labels_permuted = all_codebook_labels.permute(0, 2, 1)
        filtered_codebook_labels = all_codebook_labels_permuted[semantic_mask]
        semantic_loss = F.cross_entropy(
            codebook_logits.reshape(-1, codebook_logits.size(-1)),
            filtered_codebook_labels.reshape(-1),
            ignore_index=-100,
        )

        loss = base_loss + self.semantic_loss_weight * semantic_loss

        if is_train and self.base_kl_weight > 0:
            with torch.no_grad():
                self._set_lora_scaling(0.0)
                base_outputs = self.model(
                    inp=batch["inputs"],
                    key_padding_mask=batch["attention_masks"],
                    labels=batch["labels"],
                )
                self._set_lora_scaling(1.0)

            token_mask = (labels[:, 0] != -100).float()
            kl_token = self._reverse_kl(
                token_logits, base_outputs.token_logits, token_mask
            )
            cb_mask = (filtered_codebook_labels != -100).float()
            kl_codebook = self._reverse_kl(
                codebook_logits, base_outputs.codebook_logits, cb_mask
            )
            kl = kl_token + self.semantic_loss_weight * kl_codebook
            loss = loss + self.base_kl_weight * kl

            self.log(
                f"{stage}/base_kl",
                kl,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )

        self.log(
            f"{stage}/loss",
            loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
        )

        self.log(
            f"{stage}/base_loss",
            base_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        self.log(
            f"{stage}/semantic_loss",
            semantic_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        # Top-5 accuracy
        accuracy = self.get_accuracy(codebook_logits, filtered_codebook_labels)
        self.log(
            f"{stage}/top_5_accuracy",
            accuracy,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
        )

        return loss

    def get_accuracy(self, logits, labels):
        mask = (labels != -100) & (labels != CODEBOOK_PAD_TOKEN_ID)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        _, indices = logits.topk(5, dim=-1)
        correct = indices.eq(labels.unsqueeze(-1))
        correct[~mask] = 0
        correct = correct.sum()
        accuracy = correct / mask.sum()

        return accuracy

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
