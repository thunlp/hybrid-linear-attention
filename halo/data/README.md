## How are Datasets Loaded and Processed?

Basically, we use `load_dataset(..., streaming=True)` from the 🤗 Datasets library. Then we call the `map` method to tokenized each document in lazy mode. The token IDs are concatenated (deliminted by the `eos_token_id` argument) and then split into the chunks based on the `max_len` argument. The remainder chunk is discarded.

The return value of `get_data` is an IterableDataset, and each item fetched from the dataset is a dictionary as follows:

```python
{
    'input_ids': input_ids,  # (max_len)
    'labels': labels,  # (max_len)
}
```

> It is more common to see `max_len - 1`, because people usually split into chunks of `max_len` tokens, and then split into inputs and labels by shifting one token. We don't do that here.


## How to Add New Datasets?

In `data/__init__.py`, add your dataset to the `get_data` function. The returned value should be an IterableDataset from the 🤗 Datasets library, such as:

In general, you need to just copy `slimpj.py` and modify the following two variables

```python
def build_dataset(tokenizer, max_len, **kwargs):
    # ...
    text_column_name = 'text'
    col_names = ['text', 'meta']
    # ...
```
