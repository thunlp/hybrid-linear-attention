from pathlib import Path
from typing import Union, Optional, List, Dict

import numpy as np
from torch import Tensor
import torch
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer


def build_dataset(
    tokenizer_name: str,
    data_dir: Union[str, Path] = "/home/test/test07/data/dclm-baseline-1.0",
    streaming: bool = True,
    n_workers: int = 8,
    overwrite_cache: bool = False,
    token_ids_only: bool = True,
    max_len: int = 512,
    eos_token_id: Optional[int] = None,
    shift_labels: bool = False,
    is_seq2seq: bool = False,
    return_list: bool = False,
    **kwargs,
) -> IterableDataset:
    """
    Returns an iterable of batches of token IDs.

    This will use `load_dataset` from the HuggingFace Datasets library to load the
    data from `data_dir`, tokenize each example, concatenate the input IDs, add an
    EOS token ID at the end of each sequence, then split into chunks of `max_len`
    tokens, and return a tensor of (batch_size, max_len).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    if is_seq2seq:
        shift_labels = False

    assert eos_token_id is not None

    # print(f"Loading data from {data_dir}")
    raw_dataset = load_dataset(
        data_dir,
        split='train',
        streaming=streaming,
    )
    # print("Dataset loaded")

    text_column_name = "text"
    col_names = [
        'bff_contained_ngram_count_before_dedupe',
        'language_id_whole_page_fasttext',
        'metadata',
        'previous_word_count',
        'text',
        'url',
        'warcinfo',
        'fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob',
        # "text",
        # "id",
        # "dump",
        # "url",
        # "file_path",
        # "language",
        # "language_score",
        # "token_count",
        # "score",
        # "int_score",
    ]

    # Tokenize in streaming mode
    def process_fn(examples: dict) -> Dict[str, Tensor | List[int]]:
        """
        A process function to use with `Dataset.map`. It tokenizes
        texts in the batch, concatenate them, and split into chunks
        with `max_len` tokens (discarding the last chunk if
        incomplete).
        """
        texts: List[str] = examples[text_column_name]
        encodings = tokenizer(
            texts, max_length=10**6, truncation=True, return_tensors="np"
        )

        # Append EOS token
        orig_input_ids: np.ndarray = encodings["input_ids"]
        batch_ids = [np.append(ids, eos_token_id) for ids in orig_input_ids]
        concat_ids = np.concatenate(batch_ids)
        total_len = concat_ids.shape[0]
        if shift_labels:
            chunk_len = max_len + 1  # The input IDs with be chunk_len - 1
        else:
            chunk_len = max_len

        # Rounded down to multiple of chunk_len.
        # So the last remainder chunk is discarded.
        total_len = total_len // chunk_len * chunk_len
        n_chunks = total_len // chunk_len

        if shift_labels:
            input_ids = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
            labels = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
        else:
            input_ids = np.zeros((n_chunks, chunk_len), dtype=np.int64)
            labels = np.zeros((n_chunks, chunk_len), dtype=np.int64)

        for i in range(n_chunks):
            this_chunk = concat_ids[i * chunk_len : (i + 1) * chunk_len]
            if shift_labels:
                # Next token prediction with teacher forcing
                input_ids[i] = this_chunk[:-1]
                labels[i] = this_chunk[1:]
            else:
                input_ids[i] = this_chunk
                labels[i] = this_chunk

        # (batch_size, max_len)
        input_ids: Tensor = torch.from_numpy(input_ids)
        labels: Tensor = torch.from_numpy(labels)
        bsz = input_ids.shape[0]
        position_ids: Tensor = torch.arange(input_ids.shape[1]).repeat(bsz, 1)

        if is_seq2seq:
            assert (input_ids == labels).all()
            # For seq2seq, we need two different set of input IDs,
            # one for the encoder, one for the decoder.
            # We will evenly split into two halfs for now.
            split = input_ids.shape[1] // 2
            input_ids = input_ids[:, :split]
            decoder_input_ids = input_ids[:, split:]
            return {
                "input_ids": input_ids,
                "decoder_input_ids": decoder_input_ids,
            }

        if return_list:
            input_ids = input_ids.tolist()
            labels = labels.tolist()
            position_ids = position_ids.tolist()

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }
        return batch

    if streaming:
        # Streaming dataset does not support multi-processing yet.
        tokenized_dataset = raw_dataset.map(
            process_fn,
            batched=True,
            remove_columns=col_names if token_ids_only else [],
        )
    else:
        tokenized_dataset = raw_dataset.map(
            process_fn,
            batched=True,
            num_proc=n_workers,
            remove_columns=col_names if token_ids_only else [],
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return tokenized_dataset
