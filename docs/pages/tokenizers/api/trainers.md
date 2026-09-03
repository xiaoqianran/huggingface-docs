# Trainers

## BpeTrainer[[tokenizers.trainers.BpeTrainer]]

#### tokenizers.trainers.BpeTrainer[[tokenizers.trainers.BpeTrainer]]

```python
tokenizers.trainers.BpeTrainer(vocab_size = 30000, min_frequency = 0, show_progress = True, progress_format = 'indicatif', special_tokens = [], limit_alphabet = None, initial_alphabet = [], continuing_subword_prefix = None, end_of_word_suffix = None, max_token_length = None, words = {})
```

**Parameters:**

vocab_size (`int`, *optional*) : The size of the final vocabulary, including all tokens and alphabet. 

min_frequency (`int`, *optional*) : The minimum frequency a pair should have in order to be merged. 

show_progress (`bool`, *optional*) : Whether to show progress bars while training. 

special_tokens (`List[Union[str, AddedToken]]`, *optional*) : A list of special tokens the model should know of. 

limit_alphabet (`int`, *optional*) : The maximum different characters to keep in the alphabet. 

initial_alphabet (`List[str]`, *optional*) : A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept. 

continuing_subword_prefix (`str`, *optional*) : A prefix to be used for every subword that is not a beginning-of-word. 

end_of_word_suffix (`str`, *optional*) : A suffix to be used for every subword that is a end-of-word. 

max_token_length (`int`, *optional*) : Prevents creating tokens longer than the specified size. This can help with reducing polluting your vocabulary with highly repetitive tokens like *======* for wikipedia

Trainer capable of training a BPE model

Example:

```python
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> trainer = BpeTrainer(
...     vocab_size=30000,
...     special_tokens=["&amp;lt;unk>", "<s>", "</s>"],
...     min_frequency=2,
... )
>>> tokenizer = Tokenizer(BPE())
>>> tokenizer.train(["path/to/corpus.txt"], trainer)
```

#### get_word_count[[tokenizers.trainers.BpeTrainer.get_word_count]]

```python
get_word_count()
```

Get the number of unique words after feeding the corpus

## ParityBpeTrainer[[tokenizers.trainers.ParityBpeTrainer]]

#### tokenizers.trainers.ParityBpeTrainer[[tokenizers.trainers.ParityBpeTrainer]]

```python
tokenizers.trainers.ParityBpeTrainer(num_merges = 32000, variant = 'base', min_frequency = 0, ratio = None, global_merges = 0, window_size = 100, alpha = 2.0, total_symbols = False, special_tokens = None, show_progress = True, limit_alphabet = None, initial_alphabet = None, continuing_subword_prefix = None, end_of_word_suffix = None, max_token_length = None)
```

**Parameters:**

num_merges (`int`, *optional*) : Number of BPE merge operations to perform. Defaults to `32000`. 

variant (`str`, *optional*) : Algorithm variant: `"base"` (default) or `"window"` (moving-window balancing). 

min_frequency (`int`, *optional*) : Minimum pair frequency to merge. Defaults to `0`. 

ratio (`List[float]`, *optional*) : Target compression rate per language, one entry per training iterator. The trainer selects the language with the lowest `compression_rate / ratio`, so raising one language's ratio gives it more merges; only the values relative to each other matter. Rates are counted in the units the pre-tokenizer emits, bytes under [ByteLevel](/docs/tokenizers/v0.23.2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel), so equal ratios do not give equal tokenization across scripts: Devanagari takes about 2.5x the bytes of Latin script for the same content. Set each ratio proportional to the language's average length on parallel text instead. Defaults to `None`. 

global_merges (`int`, *optional*) : Number of initial standard BPE merges before switching to parity mode. Defaults to `0`. 

window_size (`int`, *optional*) : Window size for the `"window"` variant. Defaults to `100`. 

alpha (`float`, *optional*) : Alpha parameter for the `"window"` variant. Defaults to `2.0`. 

total_symbols (`bool`, *optional*) : If True, subtract unique character count from `num_merges`. Defaults to `False`.

Trainer for parity-aware BPE that ensures cross-lingual fairness in tokenization.

Unlike standard BPE, this trainer takes one Python iterator per language and
balances merge operations across languages using a development set or target
compression ratios. The single training entry point is
`train_from_iterator`, the multi-corpus analogue of
`tokenizers.Tokenizer.train_from_iterator()`. Balancing requires either
`dev_iterators` (ideally parallel across languages) or `ratio`; given
neither, selection follows the per-language training totals, which the
highest-data language dominates, so the result stays close to plain BPE. When
both are given the dev set takes precedence. A runnable version that obtains
per-language corpora and a parallel dev set is in
`examples/train_parity_bpe.py`.

Example:

```python
>>> from tokenizers import Tokenizer, pre_tokenizers
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import ParityBpeTrainer
>>> tokenizer = Tokenizer(BPE())
>>> tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
>>> trainer = ParityBpeTrainer(num_merges=32000, variant="base")
>>> # balance against a parallel dev set, the same sentences in every language
>>> trainer.train_from_iterator(
...     tokenizer,
...     train_iterators=[english_lines, hindi_lines],
...     dev_iterators=[english_dev, hindi_dev],
... )
>>> # or against target rates, which need no dev data
>>> trainer.train_from_iterator(
...     tokenizer,
...     train_iterators=[english_lines, hindi_lines],
...     ratio=[1.0, 2.57],
... )
```

#### train_from_iterator[[tokenizers.trainers.ParityBpeTrainer.train_from_iterator]]

```python
train_from_iterator(tokenizer, train_iterators, dev_iterators = None, ratio = None)
```

**Parameters:**

tokenizer ([Tokenizer](/docs/tokenizers/v0.23.2/en/api/tokenizer#tokenizers.Tokenizer)) : A tokenizer instance to train. Its pre-tokenizer (and optionally normalizer) should already be configured. 

train_iterators (`List[Iterator]`) : One Python iterator per language, each yielding `str` or `List[str]`. 

dev_iterators (`List[Iterator]`, *optional*) : One Python iterator per language, used to drive parity-aware language selection. Must have the same length as `train_iterators`. 

ratio (`List[float]`, *optional*) : Target compression rates per language, an alternative to `dev_iterators` and ignored when one is supplied. See the class docstring for how to choose the values.

Train a user-configured tokenizer with parity-aware BPE from per-language
Python iterators.

Each entry of `train_iterators` (and optionally `dev_iterators`) is a
Python iterator yielding strings (or batches / lists of strings) for one
language. This is the multi-corpus analogue of
`train_from_iterator()`: file I/O happens in
Python, so users can pull data from plain text, parquet (via `pyarrow`),
`datasets`, etc.

## UnigramTrainer[[tokenizers.trainers.UnigramTrainer]]

#### tokenizers.trainers.UnigramTrainer[[tokenizers.trainers.UnigramTrainer]]

```python
tokenizers.trainers.UnigramTrainer(vocab_size = 8000, show_progress = True, special_tokens = [], initial_alphabet = [], shrinking_factor = 0.75, unk_token = None, max_piece_length = 16, n_sub_iterations = 2)
```

**Parameters:**

vocab_size (`int`) : The size of the final vocabulary, including all tokens and alphabet. 

show_progress (`bool`) : Whether to show progress bars while training. 

special_tokens (`List[Union[str, AddedToken]]`) : A list of special tokens the model should know of. 

initial_alphabet (`List[str]`) : A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept. 

shrinking_factor (`float`) : The shrinking factor used at each step of the training to prune the vocabulary. 

unk_token (`str`) : The token used for out-of-vocabulary tokens. 

max_piece_length (`int`) : The maximum length of a given token. 

n_sub_iterations (`int`) : The number of iterations of the EM algorithm to perform before pruning the vocabulary.

Trainer capable of training a Unigram model

Example:

```python
>>> from tokenizers.models import Unigram
>>> from tokenizers.trainers import UnigramTrainer
>>> trainer = UnigramTrainer(
...     vocab_size=8000,
...     special_tokens=["&amp;lt;unk>", "<s>", "</s>"],
...     unk_token="REDACTED",
... )
>>> tokenizer = Tokenizer(Unigram())
>>> tokenizer.train(["path/to/corpus.txt"], trainer)
```

## WordLevelTrainer[[tokenizers.trainers.WordLevelTrainer]]

#### tokenizers.trainers.WordLevelTrainer[[tokenizers.trainers.WordLevelTrainer]]

```python
tokenizers.trainers.WordLevelTrainer(vocab_size = 30000, min_frequency = 0, show_progress = True, special_tokens = [])
```

**Parameters:**

vocab_size (`int`, *optional*) : The size of the final vocabulary, including all tokens and alphabet. 

min_frequency (`int`, *optional*) : The minimum frequency a pair should have in order to be merged. 

show_progress (`bool`, *optional*) : Whether to show progress bars while training. 

special_tokens (`List[Union[str, AddedToken]]`) : A list of special tokens the model should know of.

Trainer capable of training a WordLevel model

Example:

```python
>>> from tokenizers.models import WordLevel
>>> from tokenizers.trainers import WordLevelTrainer
>>> trainer = WordLevelTrainer(
...     vocab_size=10000,
...     special_tokens=["&amp;lt;unk>"],
...     min_frequency=1,
... )
>>> tokenizer = Tokenizer(WordLevel(unk_token="REDACTED"))
>>> tokenizer.train(["path/to/corpus.txt"], trainer)
```

## WordPieceTrainer[[tokenizers.trainers.WordPieceTrainer]]

#### tokenizers.trainers.WordPieceTrainer[[tokenizers.trainers.WordPieceTrainer]]

```python
tokenizers.trainers.WordPieceTrainer(vocab_size = 30000, min_frequency = 0, show_progress = True, special_tokens = [], limit_alphabet = None, initial_alphabet = [], continuing_subword_prefix = '##', end_of_word_suffix = None)
```

**Parameters:**

vocab_size (`int`, *optional*) : The size of the final vocabulary, including all tokens and alphabet. 

min_frequency (`int`, *optional*) : The minimum frequency a pair should have in order to be merged. 

show_progress (`bool`, *optional*) : Whether to show progress bars while training. 

special_tokens (`List[Union[str, AddedToken]]`, *optional*) : A list of special tokens the model should know of. 

limit_alphabet (`int`, *optional*) : The maximum different characters to keep in the alphabet. 

initial_alphabet (`List[str]`, *optional*) : A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept. 

continuing_subword_prefix (`str`, *optional*) : A prefix to be used for every subword that is not a beginning-of-word. 

end_of_word_suffix (`str`, *optional*) : A suffix to be used for every subword that is a end-of-word.

Trainer capable of training a WordPiece model

Example:

```python
>>> from tokenizers.models import WordPiece
>>> from tokenizers.trainers import WordPieceTrainer
>>> trainer = WordPieceTrainer(
...     vocab_size=30000,
...     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
... )
>>> tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
>>> tokenizer.train(["path/to/corpus.txt"], trainer)
```

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Visualizer
https://huggingface.co/docs/tokenizers/v0.23.2/api/visualizer.md
