import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from random import Random
from typing import Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets.download.streaming_download_manager import xopen
from huggingface_hub import HfApi
from lightning import LightningDataModule
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from fish_speech.content_sequence import ContentSequence, TextPart, VQPart

CODEBOOK_PAD_TOKEN_ID = 0

from fish_speech.datasets.protos.text_data_pb2 import SampledData
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.text.clean import clean_text
from fish_speech.tokenizer import FishTokenizer
from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand

log = RankedLogger(__name__, rank_zero_only=True)


def split_by_rank_worker(files):
    # We need to know the total number of devices
    # to split the data properly

    total_devices = 1
    if is_initialized():
        total_devices = get_world_size()

    worker_info = get_worker_info()
    if worker_info is not None:
        total_devices *= worker_info.num_workers

    if len(files) < total_devices:
        # Repeat the files N times to match the number of devices
        files = files * (total_devices // len(files) + 1)

    # DDP
    if is_initialized():
        files = files[get_rank() :: get_world_size()]

    # Split by worker
    if worker_info is not None:
        files = files[worker_info.id :: worker_info.num_workers]

    return files


class AutoTextSemanticInstructionIterableDataset(IterableDataset):
    """
    Auto Augment Dataset by Speaker

    1. Random concatenate multiple sentences from the same speaker to form a longer sentence
    2. Automatically normalize the text

    For interactive mode, we use the following format (multiple sequences):
    <s> [INST] [SPK: speaker] text [/INST] ... [INST] text [/INST] </s>

    For non-interactive mode, we use the following format (one long sequence):
    <s> [INST] text [/INST] ... </s>
    """

    def __init__(
        self,
        proto_files: list[str],
        seed: int = 42,
        interactive_prob: float = 0.5,
        max_length: int = 1024,
        tokenizer: FishTokenizer = None,
        use_speaker: bool | float = True,
        causal: bool = True,
        num_codebooks: Optional[int] = None,
        skip_text_prob: float = 0.0,
    ):
        """
        Args:
            proto_files: proto buf files if using local data
            seed: random seed
            interactive_prob: probability to use interactive mode
            max_length: max length of the text
            tokenizer: tokenizer
            use_speaker: include speaker information in the prompt
            causal: use causal sampling when using local data, disable will lead to random sampling
            num_codebooks: number of codebooks, if None, it will be automatically detected
            skip_text_prob: probability to skip the text (audio only), this only applies to interactive mode
        """

        super().__init__()

        assert 0 <= interactive_prob <= 1, "interactive_prob must be in [0, 1]"

        self.seed = seed
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.interactive_prob = interactive_prob
        self.use_speaker = use_speaker
        self.proto_files = proto_files
        self.causal = causal
        self.num_codebooks = num_codebooks
        self.skip_text_prob = skip_text_prob

        self.groups = None

    def __iter__(self):
        while True:
            yield self.augment()

    def init_mock_data_server(self):
        if self.groups is not None:
            return

        # Expand the proto files
        expanded_proto_files = []
        for filename in self.proto_files:
            for i in braceexpand(filename):
                i = Path(i)
                if i.is_file():
                    expanded_proto_files.append(i)
                elif i.is_dir():
                    expanded_proto_files.extend(i.rglob("*.proto"))
                    expanded_proto_files.extend(i.rglob("*.protos"))
                else:
                    raise ValueError(f"{i} is not a file or directory")

        expanded_proto_files = sorted(expanded_proto_files)
        Random(self.seed).shuffle(expanded_proto_files)

        self.groups = []
        shard_proto_files = split_by_rank_worker(expanded_proto_files)
        log.info(
            f"Reading {len(shard_proto_files)} / {len(expanded_proto_files)} files"
        )

        count = 0
        for filename in shard_proto_files:
            with open(filename, "rb") as f:
                for text_data in read_pb_stream(f):
                    self.groups.append(text_data)
                    count += 1

        log.info(f"Read total {count} groups of data")

        # Shuffle the lines
        Random(self.seed).shuffle(self.groups)
        self.group_weights = [len(i.sentences) for i in self.groups]

    def sample_data(self):
        if self.groups is None:
            self.init_mock_data_server()

        # Shuffle unique lines, estimate that each sample is at least 20 tokens
        num_samples = self.max_length // 20

        # choice group based on their number of samples
        group = random.choices(self.groups, weights=self.group_weights, k=1)[0]

        if self.causal:
            # Sample in order
            if num_samples >= len(group.sentences):
                samples = group.sentences
            else:
                begin = random.randint(0, len(group.sentences) - num_samples)
                samples = group.sentences[begin : begin + num_samples]
        else:
            samples = random.choices(
                group.sentences, k=min(num_samples, len(group.sentences))
            )

        return SampledData(
            source=group.source,
            name=group.name,
            samples=samples,
        )

    def pack_sentences(
        self,
        sentences: list[str],
        semantics: list,
        # speaker: Optional[str] = None, # speaker is now handled by tokens
        skip_text: bool = False,
    ):

        seq = ContentSequence()

        seq.append(TextPart(text="Speak out the provided text."))

        # User's turn
        cated_sentences = " ".join(sentences)
        if skip_text:
            cated_sentences = "<|skip_text|>"

        seq.append(
            TextPart(text=f"<|speaker:user|> {cated_sentences}"),
            add_end=True,
        )

        # Assistant's turn
        vq_codes = [x.values for x in semantics[0]]
        vq_codes_tensor = torch.tensor(vq_codes).to(torch.int32)

        # 将 cal_loss=True 直接关联到 VQPart 上，这比之前更精确
        vq_part = VQPart(codes=vq_codes_tensor, cal_loss=True)

        # 将多个 parts 一起添加，最后也加上 <|im_end|>
        seq.append(
            [TextPart(text="<|speaker:assistant|> <|voice|>"), vq_part],
            add_end=True,
        )

        encoded = seq.encode(
            tokenizer=self.tokenizer,
        )

        num_codebooks = (
            len(semantics[0]) if self.num_codebooks is None else self.num_codebooks
        )

        tokens_raw = encoded.tokens
        tokens = torch.zeros((num_codebooks + 1, len(tokens_raw)), dtype=torch.int)
        tokens[0] = tokens_raw

        vq_parts = encoded.vq_parts
        vq_parts = [part.to(tokens.device) for part in vq_parts]
        vq_parts = torch.cat(vq_parts, dim=1)
        tokens[1:, encoded.vq_mask_tokens] = vq_parts

        labels_raw = encoded.labels
        labels = torch.full((num_codebooks + 1, len(labels_raw)), -100, dtype=torch.int)
        labels[0, :] = labels_raw
        labels[1:, encoded.vq_mask_labels] = vq_parts
        labels[1:, -1:] = CODEBOOK_PAD_TOKEN_ID

        tokens = tokens.long()
        labels = labels.long()

        # Verify the padding is correct, and the last token is eos
        assert (tokens[1:, ~(encoded.vq_mask_tokens)] == CODEBOOK_PAD_TOKEN_ID).all()
        assert (labels[1:, -1:] == CODEBOOK_PAD_TOKEN_ID).all()

        return tokens, labels

    def augment(self):
        response = self.sample_data()
        if len(response.samples) == 0:
            # Invalid group
            return None

        samples = list(response.samples)
        all_tokens, all_labels = [], []

        while len(samples) > 0:
            sentence = samples.pop(0)
            text = clean_text(random.choice(sentence.texts))

            tokens, labels = self.pack_sentences(
                sentences=[text],
                semantics=[sentence.semantics],
                # speaker=response.name if use_speaker else None,
                skip_text=random.random() < self.skip_text_prob,
            )

            all_tokens.append(tokens)
            all_labels.append(labels)

        tokens = torch.cat(all_tokens, dim=1)
        labels = torch.cat(all_labels, dim=1)

        # Verify that the length is correct
        assert tokens.size(1) == labels.size(1), f"{tokens.size(1)} != {labels.size(1)}"

        data = {"tokens": tokens, "labels": labels}

        return data


class AutoTextSemanticInstructionDataset(Dataset):
    """
    Auto Augment Dataset by Speaker

    1. Random concatenate multiple sentences from the same speaker to form a longer sentence
    2. Automatically normalize the text

    For interactive mode, we use the following format (multiple sequences):
    <s> [INST] [SPK: speaker] text [/INST] ... [INST] text [/INST] </s>

    For non-interactive mode, we use the following format (one long sequence):
    <s> [INST] text [/INST] ... </s>
    """

    def __init__(
        self,
        proto_files: list[str],
        seed: int = 42,
        interactive_prob: float = 0.5,
        max_length: int = 1024,
        tokenizer: FishTokenizer = None,
        use_speaker: bool | float = True,
        causal: bool = True,
        num_codebooks: Optional[int] = None,
        skip_text_prob: float = 0.0,
    ):
        """
        Args:
            proto_files: proto buf files if using local data
            seed: random seed
            interactive_prob: probability to use interactive mode
            max_length: max length of the text
            tokenizer: tokenizer
            use_speaker: include speaker information in the prompt
            causal: use causal sampling when using local data, disable will lead to random sampling
            num_codebooks: number of codebooks, if None, it will be automatically detected
            skip_text_prob: probability to skip the text (audio only), this only applies to interactive mode
        """
        super().__init__()

        assert 0 <= interactive_prob <= 1, "interactive_prob must be in [0, 1]"

        self.seed = seed
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.interactive_prob = interactive_prob
        self.use_speaker = use_speaker
        self.proto_files = proto_files
        self.causal = causal
        self.num_codebooks = num_codebooks
        self.skip_text_prob = skip_text_prob

        self.data = []
        self._init_data()

    def _init_data(self):
        expanded_proto_files = []
        for filename in self.proto_files:
            for i in braceexpand(filename):
                i = Path(i)
                if i.is_file():
                    expanded_proto_files.append(i)
                elif i.is_dir():
                    expanded_proto_files.extend(i.rglob("*.proto"))
                    expanded_proto_files.extend(i.rglob("*.protos"))
                else:
                    raise ValueError(f"{i} is not a file or directory")

        expanded_proto_files = sorted(expanded_proto_files)
        Random(self.seed).shuffle(expanded_proto_files)

        groups = []
        shard_proto_files = split_by_rank_worker(expanded_proto_files)
        log.info(
            f"Reading {len(shard_proto_files)} / {len(expanded_proto_files)} files"
        )

        count = 0
        for filename in shard_proto_files:
            with open(filename, "rb") as f:
                for text_data in read_pb_stream(f):
                    groups.append(text_data)
                    count += 1

        log.info(f"Read total {count} groups of data")

        for group in groups:
            if len(group.sentences) == 0:
                continue

            samples = list(group.sentences)
            for sentence in samples:
                text = clean_text(random.choice(sentence.texts))

                tokens, labels = self.pack_sentences(
                    sentences=[text],
                    semantics=[sentence.semantics],
                    skip_text=random.random() < self.skip_text_prob,
                )

                self.data.append({"tokens": tokens, "labels": labels})

        random.Random(self.seed).shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_sentences(
        self,
        sentences: list[str],
        semantics: list,
        skip_text: bool = False,
    ):
        messages = [
            Message(
                role="system",
                parts=[TextPart(text="Speak out the provided text.")],
            )
        ]

        cated_sentences = " ".join(sentences)
        if skip_text:
            cated_sentences = "<|skip_text|>"

        messages.append(
            Message(
                role="user",
                parts=[TextPart(text=cated_sentences)],
            )
        )

        vq_codes = [x.values for x in semantics[0]]
        vq_codes_tensor = torch.tensor(vq_codes).to(torch.int32)
        vqpart = VQPart(codes=vq_codes_tensor)
        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>"), vqpart],
                cal_loss=True,
            )
        )

        num_codebooks = (
            len(semantics[0]) if self.num_codebooks is None else self.num_codebooks
        )

        conversation = Conversation(messages=messages)
        encoded = conversation.encode(
            tokenizer=self.tokenizer,
        )

        tokens_raw = encoded.tokens
        tokens = torch.zeros((num_codebooks + 1, len(tokens_raw)), dtype=torch.int)
        tokens[0] = tokens_raw

        vq_parts = encoded.vq_parts
        vq_parts = [part.to(tokens.device) for part in vq_parts]
        vq_parts = torch.cat(vq_parts, dim=1)
        tokens[1:, encoded.vq_mask_tokens] = vq_parts

        labels_raw = encoded.labels
        labels = torch.full((num_codebooks + 1, len(labels_raw)), -100, dtype=torch.int)
        labels[0, :] = labels_raw
        labels[1:, encoded.vq_mask_labels] = vq_parts
        labels[1:, -1:] = CODEBOOK_PAD_TOKEN_ID

        tokens = tokens.long()
        labels = labels.long()

        assert (tokens[1:, ~(encoded.vq_mask_tokens)] == CODEBOOK_PAD_TOKEN_ID).all()
        assert (labels[1:, -1:] == CODEBOOK_PAD_TOKEN_ID).all()

        return tokens, labels


class InferenceMatchedIterableDataset(AutoTextSemanticInstructionIterableDataset):
    """Training samples in the EXACT prompt format used by generate_long().

    The stock fine-tuning format ("Speak out the provided text.<|speaker:user|>
    ...") never shows the model the inference prompt — neither the
    "convert the provided text to speech" system message, the <|im_start|>
    role structure, nor reference (voice-clone) conditioning. Fine-tuning on it
    teaches the model a format that inference never uses.

    This dataset instead mirrors fish_speech/models/text2semantic/inference.py:
    with probability ``ref_prob`` (and >= 2 utterances in the sampled speaker
    group) one utterance becomes the voice-cloning reference in the system
    message and the rest become user/assistant turns; otherwise the
    unconditioned system message is used. Loss is calculated only on the
    assistant turns (semantic tokens + <|im_end|>), matching what generation
    must produce. This is also how Qwen3-TTS SFT works: every row is
    conditioned on a same-speaker reference utterance.
    """

    def __init__(
        self, *args, ref_prob: float = 0.8, ipa_token_map: str | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert 0 <= ref_prob <= 1, "ref_prob must be in [0, 1]"
        self.ref_prob = ref_prob
        # Atomic IPA tokens: convert every text through the map so IPA "j"
        # etc. become dedicated tokens instead of English-biased BPE pieces.
        self._ipa_map = None
        if ipa_token_map is not None:
            from fish_speech.text.ipa_tokens import load_token_map

            self._ipa_map = load_token_map(ipa_token_map)

    def _maybe_convert(self, text: str) -> str:
        if self._ipa_map is None:
            return text
        from fish_speech.text.ipa_tokens import convert_ipa

        return convert_ipa(text, self._ipa_map)

    @staticmethod
    def _codes_tensor(sentence) -> torch.Tensor:
        return torch.tensor([x.values for x in sentence.semantics], dtype=torch.int32)

    def _build_conversation(self, ref, targets):
        from fish_speech.conversation import Conversation, Message

        conversation = Conversation()

        # System message: byte-identical to generate_long()
        if ref is not None:
            ref_text, ref_codes = ref
            system_parts = [
                TextPart(
                    text="convert the provided text to speech reference to the following:\n\nText:\n",
                    cal_loss=False,
                ),
                TextPart(text=f"<|speaker:0|>{ref_text}", cal_loss=False),
                TextPart(text="\n\nSpeech:\n", cal_loss=False),
                VQPart(codes=ref_codes, cal_loss=False),
            ]
        else:
            system_parts = [
                TextPart(text="convert the provided text to speech", cal_loss=False)
            ]

        conversation.append(
            Message(
                role="system",
                parts=system_parts,
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )

        for text, codes in targets:
            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )
            conversation.append(
                Message(
                    role="assistant",
                    parts=[VQPart(codes=codes, cal_loss=True)],
                    cal_loss=True,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=True,
                )
            )

        return conversation

    def _encode(self, ref, targets):
        conversation = self._build_conversation(ref, targets)
        # Not Conversation.encode(): it forwards a max_length kwarg that
        # ContentSequence.encode() does not accept.
        encoded = conversation.to_content_sequence().encode(self.tokenizer)

        num_codebooks = (
            targets[0][1].size(0) if self.num_codebooks is None else self.num_codebooks
        )

        tokens_raw = encoded.tokens
        tokens = torch.zeros((num_codebooks + 1, len(tokens_raw)), dtype=torch.int)
        tokens[0] = tokens_raw

        vq_parts = encoded.vq_parts
        vq_parts = [part.to(tokens.device) for part in vq_parts]
        vq_parts = torch.cat(vq_parts, dim=1)
        tokens[1:, encoded.vq_mask_tokens] = vq_parts

        labels_raw = encoded.labels
        labels = torch.full((num_codebooks + 1, len(labels_raw)), -100, dtype=torch.int)
        labels[0, :] = labels_raw
        labels[1:, encoded.vq_mask_labels] = vq_parts

        tokens = tokens.long()
        labels = labels.long()

        assert (tokens[1:, ~(encoded.vq_mask_tokens)] == CODEBOOK_PAD_TOKEN_ID).all()

        return tokens, labels

    def augment(self):
        response = self.sample_data()
        if len(response.samples) == 0:
            return None

        samples = list(response.samples)
        random.shuffle(samples)

        ref = None
        if len(samples) >= 2 and random.random() < self.ref_prob:
            sentence = samples.pop()
            ref = (
                self._maybe_convert(clean_text(random.choice(sentence.texts))),
                self._codes_tensor(sentence),
            )

        # Add target turns until the encoded sequence would exceed max_length.
        targets = []
        tokens = labels = None
        for sentence in samples:
            targets.append(
                (
                    self._maybe_convert(clean_text(random.choice(sentence.texts))),
                    self._codes_tensor(sentence),
                )
            )
            new_tokens, new_labels = self._encode(ref, targets)
            if tokens is not None and new_tokens.size(1) > self.max_length:
                targets.pop()
                break
            tokens, labels = new_tokens, new_labels
            if tokens.size(1) >= self.max_length:
                break

        if tokens is None:
            return None

        return {"tokens": tokens, "labels": labels}


class InterleaveDataset(IterableDataset):
    def __init__(
        self,
        datasets: list[IterableDataset],
        probabilities: list[float],
        seed: int = 42,
    ):
        super().__init__()

        self.datasets = datasets
        self.probabilities = probabilities
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        dataset_iterators = [iter(dataset) for dataset in self.datasets]

        while True:
            # Random choice one
            dataset_idx = rng.choice(len(self.datasets), p=self.probabilities)
            dataset_iterator = dataset_iterators[dataset_idx]

            try:
                yield next(dataset_iterator)
            except StopIteration:
                # Exhausted, create a new iterator
                dataset_iterators[dataset_idx] = iter(self.datasets[dataset_idx])
                yield next(dataset_iterators[dataset_idx])


@dataclass
class TextDataCollator:
    tokenizer: FishTokenizer
    max_length: int = 1024

    def __call__(self, examples):
        if "negative_tokens" in examples:
            positive_examples = []
            negative_examples = []

            for i in examples:
                positive_examples.append(
                    {
                        "tokens": i["tokens"],
                        "labels": i["labels"],
                    }
                )
                negative_examples.append(
                    {
                        "tokens": i["negative_tokens"],
                        "labels": i["negative_labels"],
                    }
                )

            examples = positive_examples + negative_examples

        return self.batchify(examples)

    def batchify(self, examples, tokens_key="tokens", labels_key="labels"):
        tokens, attention_masks, labels = [], [], []

        # Calculate the max length
        max_tokens_length = 0
        for example in examples:
            max_tokens_length = max(max_tokens_length, example[tokens_key].size(1))
        max_tokens_length = min(max_tokens_length, self.max_length)

        for example in examples:
            _tokens = example[tokens_key][:, :max_tokens_length]
            _labels = example[labels_key][:, :max_tokens_length]
            _attention_mask = torch.ones((max_tokens_length,), dtype=torch.bool)
            tokens_length = _tokens.size(1)
            _attention_mask[:tokens_length] = False

            assert tokens_length == _labels.size(
                1
            ), f"{tokens_length} != {_labels.size(1)}"

            if tokens_length < max_tokens_length:
                _tokens = F.pad(
                    _tokens,
                    (0, max_tokens_length - tokens_length),
                    value=self.tokenizer.get_token_id("<|end_of_text|>"),
                )
                _tokens[1:, tokens_length:] = CODEBOOK_PAD_TOKEN_ID
                _labels = F.pad(
                    _labels, (0, max_tokens_length - _labels.size(1)), value=-100
                )

            tokens.append(_tokens)
            attention_masks.append(_attention_mask)
            labels.append(_labels)

        tokens = torch.stack(tokens, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0)

        return {
            "inputs": tokens,
            "attention_masks": attention_masks,
            "labels": labels,
        }


class SemanticDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Union[
            AutoTextSemanticInstructionDataset,
            AutoTextSemanticInstructionIterableDataset,
            InterleaveDataset,
        ],
        val_dataset: Union[
            AutoTextSemanticInstructionDataset,
            AutoTextSemanticInstructionIterableDataset,
            InterleaveDataset,
        ],
        batch_size: int = 32,
        tokenizer: FishTokenizer = None,
        max_length: int = 1024,
        num_workers: int = 4,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=TextDataCollator(self.tokenizer, self.max_length),
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=TextDataCollator(self.tokenizer, self.max_length),
            num_workers=self.num_workers,
            persistent_workers=True,
        )


if __name__ == "__main__":
    from tqdm import tqdm

    ds = AutoTextSemanticInstructionDataset(
        ["data/protos"],
        tokenizer=FishTokenizer("checkpoints/fish-speech-1.5/tokenizer.tiktoken"),
        use_speaker=False,
        interactive_prob=1.0,
        skip_text_prob=0.5,
    )

    for i in range(100):
        # Please uncomment line 235 to visualize the tokenized message
        print(ds[i])
