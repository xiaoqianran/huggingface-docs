# Trainers

## BpeTrainer[[tokenizers.trainers.BpeTrainer]]

#### tokenizers.trainers.BpeTrainer[[tokenizers.trainers.BpeTrainer]]

Trainer capable of training a BPE model

Example:

```python
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> trainer = BpeTrainer(
...     vocab_size=30000,
...     special_tokens=["&amp;lt;unk>", "", ""],
...     min_frequency=2,
... )
>>> tokenizer = Tokenizer(BPE())
>>> tokenizer.train(["path/to/corpus.txt"], trainer)
```

get_word_counttokenizers.trainers.BpeTrainer.get_word_count[]
Get the number of unique words after feeding the corpus

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

## UnigramTrainer[[tokenizers.trainers.UnigramTrainer]]

#### tokenizers.trainers.UnigramTrainer[[tokenizers.trainers.UnigramTrainer]]

Trainer capable of training a Unigram model

Example:

```python
>>> from tokenizers.models import Unigram
>>> from tokenizers.trainers import UnigramTrainer
>>> trainer = UnigramTrainer(
...     vocab_size=8000,
...     special_tokens=["&amp;lt;unk>", "", ""],
...     unk_token="&amp;lt;unk>",
... )
>>> tokenizer = Tokenizer(Unigram())
>>> tokenizer.train(["path/to/corpus.txt"], trainer)
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

## WordLevelTrainer[[tokenizers.trainers.WordLevelTrainer]]

#### tokenizers.trainers.WordLevelTrainer[[tokenizers.trainers.WordLevelTrainer]]

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
>>> tokenizer = Tokenizer(WordLevel(unk_token="&amp;lt;unk>"))
>>> tokenizer.train(["path/to/corpus.txt"], trainer)
```

**Parameters:**

vocab_size (`int`, *optional*) : The size of the final vocabulary, including all tokens and alphabet. 

min_frequency (`int`, *optional*) : The minimum frequency a pair should have in order to be merged. 

show_progress (`bool`, *optional*) : Whether to show progress bars while training. 

special_tokens (`List[Union[str, AddedToken]]`) : A list of special tokens the model should know of.

## WordPieceTrainer[[tokenizers.trainers.WordPieceTrainer]]

#### tokenizers.trainers.WordPieceTrainer[[tokenizers.trainers.WordPieceTrainer]]

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

**Parameters:**

vocab_size (`int`, *optional*) : The size of the final vocabulary, including all tokens and alphabet. 

min_frequency (`int`, *optional*) : The minimum frequency a pair should have in order to be merged. 

show_progress (`bool`, *optional*) : Whether to show progress bars while training. 

special_tokens (`List[Union[str, AddedToken]]`, *optional*) : A list of special tokens the model should know of. 

limit_alphabet (`int`, *optional*) : The maximum different characters to keep in the alphabet. 

initial_alphabet (`List[str]`, *optional*) : A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept. 

continuing_subword_prefix (`str`, *optional*) : A prefix to be used for every subword that is not a beginning-of-word. 

end_of_word_suffix (`str`, *optional*) : A suffix to be used for every subword that is a end-of-word.

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Encoding
https://huggingface.co/docs/tokenizers/v0.23.1/api/encoding.md
