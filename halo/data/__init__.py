from datasets import IterableDataset

from . import fineweb_edu_100bt
from . import dclm


def get_data(
    tokenizer_name: str = '/path/to/qwen3-0.6b',
    data_name: str = "fineweb-edu-100bt",
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
    if data_name == 'fineweb-edu-100bt':
        return fineweb_edu_100bt.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            is_seq2seq=is_seq2seq,
            **kwargs,
        )
    elif data_name == 'dclm':
        return dclm.build_dataset(
            tokenizer_name=tokenizer_name,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
        )
    else:
        raise ValueError(f"Unknown data name: {data_name}")
