from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from loguru import logger


class SampleAudioCallback(L.Callback):
    """Periodically synthesize audio from the model being trained.

    Every ``every_n_train_steps`` optimizer steps, rank 0 runs the real
    inference path (``generate_long`` + the DAC codec) on a fixed list of
    prompts and logs the waveforms to TensorBoard / W&B and to
    ``<run_dir>/samples/step_<n>/``.

    Two pieces of model state have to be handled carefully:

    * **KV caches.** ``generate`` allocates ``kv_cache`` on every attention
      block. The training forward pass calls layers without ``input_pos``,
      which crashes if a cache is still attached, so the caches are always
      torn down afterwards (see ``_teardown_caches``).
    * **LoRA merging.** ``loralib`` merges the LoRA delta into the base weight
      in ``train(False)`` and subtracts it again in ``train(True)``. In
      ``bf16-true`` that round trip is lossy, so by default this callback puts
      the model into eval *semantics* (dropout off) without triggering the
      merge — the unmerged forward path computes exactly the same result.
      Pass ``merge_lora=True`` for the conventional ``eval()``/``train()``.
    """

    def __init__(
        self,
        texts: list[str],
        codec_checkpoint_path: str,
        every_n_train_steps: int = 500,
        prompt_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 30,
        chunk_length: int = 512,
        keep_codec_loaded: bool = False,
        merge_lora: bool = False,
        save_to_disk: bool = True,
        enabled: bool = True,
    ):
        super().__init__()

        self.texts = list(texts)
        self.codec_checkpoint_path = codec_checkpoint_path
        self.every_n_train_steps = every_n_train_steps
        self.prompt_audio = prompt_audio
        self.prompt_text = prompt_text
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.chunk_length = chunk_length
        self.keep_codec_loaded = keep_codec_loaded
        self.merge_lora = merge_lora
        self.save_to_disk = save_to_disk
        self.enabled = enabled

        self._codec = None
        self._prompt_tokens = None
        self._last_sampled_step = -1

    # ------------------------------------------------------------------ hooks

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enabled or self.every_n_train_steps <= 0:
            return

        step = trainer.global_step
        if step == 0 or step % self.every_n_train_steps != 0:
            return

        # With gradient accumulation several batches share one global_step.
        if step == self._last_sampled_step:
            return
        self._last_sampled_step = step

        if not trainer.is_global_zero:
            return

        try:
            self._sample(trainer, pl_module, step)
        except Exception:  # never let sampling kill a training run
            logger.exception(f"Audio sampling failed at step {step}")

    def on_train_end(self, trainer, pl_module):
        self._release_codec()

    # ---------------------------------------------------------------- helpers

    def _sample(self, trainer, pl_module, step: int):
        from fish_speech.models.text2semantic.inference import (
            decode_one_token_ar,
            decode_to_audio,
            generate_long,
        )

        model = pl_module.model
        device = pl_module.device
        was_training = model.training

        codec = self._get_codec(device)
        prompt_tokens, prompt_text = self._get_prompt(codec, device)

        self._set_eval(model)
        try:
            for idx, text in enumerate(self.texts):
                segments = [
                    r.codes
                    for r in generate_long(
                        model=model,
                        device=device,
                        decode_one_token=decode_one_token_ar,
                        text=text,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        compile=False,
                        chunk_length=self.chunk_length,
                        prompt_text=prompt_text,
                        prompt_tokens=prompt_tokens,
                    )
                    if r.action == "sample"
                ]

                if not segments:
                    logger.warning(f"No codes generated for sample {idx} at {step}")
                    continue

                codes = torch.cat(segments, dim=1).to(device)
                wav = decode_to_audio(codes, codec).float().cpu().numpy().copy()
                self._log_audio(trainer, f"sample_{idx}", wav, codec.sample_rate, step)
                self._log_codes(trainer, f"sample_{idx}", codes, step)
        finally:
            self._teardown_caches(model)
            self._restore_train(model, was_training)
            if not self.keep_codec_loaded:
                self._release_codec()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_audio(self, trainer, name: str, wav, sample_rate: int, step: int):
        import numpy as np

        peak = float(np.abs(wav).max()) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak

        # Objective health signals, so progress can be read off a curve instead
        # of by ear. Training audio sits around rms 0.07-0.20; a model emitting
        # near-silent "ghost" audio shows up here long before it is obvious.
        rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean())) if wav.size else 0.0
        duration = len(wav) / sample_rate

        for lg in trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_audio(
                    f"samples/{name}", wav, step, sample_rate=sample_rate
                )
                lg.experiment.add_scalar(f"sample_rms/{name}", rms, step)
                lg.experiment.add_scalar(f"sample_duration/{name}", duration, step)
            elif isinstance(lg, WandbLogger):
                import wandb

                lg.experiment.log(
                    {f"samples/{name}": wandb.Audio(wav, sample_rate=sample_rate)},
                    step=step,
                )

        if self.save_to_disk:
            import soundfile as sf

            out_dir = Path(trainer.default_root_dir) / "samples" / f"step_{step:09d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_dir / f"{name}.wav"), wav, sample_rate)

        logger.info(
            f"Sample {name} @ step {step}: {duration:.2f}s rms={rms:.4f} peak={peak:.3f}"
        )

    def _log_codes(self, trainer, name: str, codes, step: int):
        """Dump the generated tokens; a near-constant row 0 means the sampler
        collapsed (audio decodes to silence) rather than the model being merely
        undertrained."""
        import numpy as np

        arr = codes.detach().cpu().numpy()
        row0 = arr[0]
        uniq = len(np.unique(row0))
        longest = 1
        run = 1
        for a, b in zip(row0[:-1], row0[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        logger.info(
            f"Codes {name} @ step {step}: {row0.size} frames, "
            f"{uniq} unique semantic tokens, longest repeat run {longest}"
        )

        for lg in trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_scalar(f"sample_unique_tokens/{name}", uniq, step)
                lg.experiment.add_scalar(f"sample_max_repeat/{name}", longest, step)

        if self.save_to_disk:
            out_dir = Path(trainer.default_root_dir) / "samples" / f"step_{step:09d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_dir / f"{name}.codes.npy"), arr)

    def _get_codec(self, device):
        if self._codec is None:
            from fish_speech.models.text2semantic.inference import load_codec_model

            self._codec = load_codec_model(self.codec_checkpoint_path, device=device)
            logger.info(f"Loaded codec for sampling from {self.codec_checkpoint_path}")

        return self._codec

    def _release_codec(self):
        self._codec = None
        self._prompt_tokens = None

    def _get_prompt(self, codec, device):
        """Encode the reference audio once, for voice-cloning style samples."""
        if self.prompt_audio is None or self.prompt_text is None:
            return None, None

        if self._prompt_tokens is None:
            from fish_speech.models.text2semantic.inference import encode_audio

            self._prompt_tokens = encode_audio(self.prompt_audio, codec, device).cpu()

        # generate_long expects lists (it calls bool(prompt_tokens), which is
        # ambiguous on a bare tensor).
        return [self._prompt_tokens], [self.prompt_text]

    @staticmethod
    def _teardown_caches(model):
        """Detach KV caches; the training forward pass cannot run with them."""
        for group in ("layers", "fast_layers"):
            for block in getattr(model, group, []) or []:
                block.attention.kv_cache = None

        # setup_caches() early-returns unless these are smaller than requested
        model.max_seq_len = -1
        model.max_batch_size = -1
        model._cache_setup_done = False

    def _set_eval(self, model):
        if self.merge_lora:
            model.eval()
            return

        # eval semantics without loralib's lossy bf16 merge/unmerge round trip
        self._was_training = {m: m.training for m in model.modules()}
        for m in model.modules():
            m.training = False

    def _restore_train(self, model, was_training: bool):
        if self.merge_lora:
            model.train(was_training)
            return

        for m, flag in self._was_training.items():
            m.training = flag
        self._was_training = {}
