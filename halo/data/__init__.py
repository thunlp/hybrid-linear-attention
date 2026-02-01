from typing import Optional
import torch

from datasets import IterableDataset

from . import slimpj
from . import slimpj_200k
from . import wikitext
from . import openwebtext
from . import fineweb_edu_100bt
from . import ultrachat_200k
from . import dclm
from . import dclm_radlads
from . import prolong
from . import entropylong
from . import litelong_nextlong


def get_data(
    tokenizer_name: str = '/path/to/qwen3-0.6b',
    data_name: str = "slimpj",
    data_path: str | None = None,
    max_len: int = 512,
    shift_labels: bool = False,
    is_seq2seq: bool = False,
    stage: int | None = None,
    **kwargs,
) -> IterableDataset:
    '''
    Will return an IterableDataset
    '''
    assert data_path is not None
    if data_name == "slimpj":
        assert data_path is not None
        return slimpj.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == "slimpj-200k":
        assert data_path is not None
        return slimpj_200k.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == "wikitext":
        return wikitext.build_dataset(
            tokenizer_name=tokenizer_name,
            max_len=max_len,
            shift=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == 'openwebtext':
        return openwebtext.build_dataset(
            tokenizer_name=tokenizer_name,
            max_len=max_len,
            shift=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == 'fineweb-edu-100bt':
        return fineweb_edu_100bt.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            is_seq2seq=is_seq2seq,
            **kwargs,
        )
    elif data_name == 'ultrachat_200k':
        return ultrachat_200k.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
        )
    elif data_name == 'dclm':
        return dclm.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
        )
    elif data_name == 'dclm-radlads':
        return dclm_radlads.build_dataset(
            tokenizer_name=tokenizer_name,
            max_len=max_len,
            stage=stage,
        )
    elif data_name == 'prolong-512k':
        return prolong.build_dataset(
            tokenizer_name=tokenizer_name,
            max_len=max_len,
        )
    elif data_name == 'long-mix':
        from datasets import interleave_datasets
        print(f"Building long-mix with {shift_labels=} and {is_seq2seq=}")
        iterable_datasets = [
            prolong.build_dataset(
                tokenizer_name=tokenizer_name,
                data_path="/home/test/test07/data/prolong-512k",
                max_len=max_len,
                shift_labels=shift_labels,
                is_seq2seq=is_seq2seq,
                return_list=True,
                tokenize_batch_size=8,
            ),
            fineweb_edu_100bt.build_dataset(
                tokenizer_name=tokenizer_name,
                data_dir="/home/test/testdata/hf_data/fineweb-edu/sample/100BT",
                max_len=max_len,
                shift_labels=shift_labels,
                is_seq2seq=is_seq2seq,
                return_list=True,
                tokenize_batch_size=1000,
            )
        ]
        # print([type(ds) for ds in iterable_datasets])
        ds = interleave_datasets(iterable_datasets, seed=0)
        # it = iter(ds)
        # eg0 = next(it)
        # print(list(eg0.keys()))
        # print(eg0['input_ids'][:100])
        # breakpoint()

        # Turn it back into tensors
        def convert_to_tensors(eg):
            for key in list(eg.keys()):
                eg[key] = torch.tensor(eg[key], dtype=torch.long)
            return eg
        ds = ds.map(convert_to_tensors)
        return ds
    elif data_name == 'entropylong_128k':
        return entropylong.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
        )
    elif data_name == 'litelong_nextlong_128k':
        return litelong_nextlong.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
        )
    elif data_name == 'litelong_nextlong_512k':
        raise ValueError("litelong_nextlong_512k is not supported yet")
    else:
        raise ValueError(f"Unknown data name: {data_name}")
