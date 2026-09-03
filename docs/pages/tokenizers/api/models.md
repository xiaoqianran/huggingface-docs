# Models

## BPE[[tokenizers.models.BPE]]

#### tokenizers.models.BPE[[tokenizers.models.BPE]]

```python
tokenizers.models.BPE(vocab = None, merges = None, cache_capacity = None, dropout = None, unk_token = None, continuing_subword_prefix = None, end_of_word_suffix = None, fuse_unk = None, byte_fallback = False, ignore_merges = False)
```

**Parameters:**

vocab (`Dict[str, int]`, *optional*) : A dictionary of string keys and their ids `&amp;lcub;"am": 0,...}` 

merges (`List[Tuple[str, str]]`, *optional*) : A list of pairs of tokens (`Tuple[str, str]`) `[("a", "b"),...]` 

cache_capacity (`int`, *optional*) : The number of words that the BPE cache can contain. The cache allows to speed-up the process by keeping the result of the merge operations for a number of words. 

dropout (`float`, *optional*) : A float between 0 and 1 that represents the BPE dropout to use. 

unk_token (`str`, *optional*) : The unknown token to be used by the model. 

continuing_subword_prefix (`str`, *optional*) : The prefix to attach to subword units that don't represent a beginning of word. 

end_of_word_suffix (`str`, *optional*) : The suffix to attach to subword units that represent an end of word. 

fuse_unk (`bool`, *optional*) : Whether to fuse any subsequent unknown tokens into a single one 

byte_fallback (`bool`, *optional*) : Whether to use spm byte-fallback trick (defaults to False) 

ignore_merges (`bool`, *optional*) : Whether or not to match tokens with the vocab before using merges.

An implementation of the BPE (Byte-Pair Encoding) algorithm

Example:

```python
>>> from tokenizers.models import BPE
>>> # Build an empty model (to be trained)
>>> model = BPE(unk_token="REDACTED")
>>> # Load from vocabulary and merges files
>>> model = BPE.from_file("vocab.json", "merges.txt")
```

#### from_file[[tokenizers.models.BPE.from_file]]

```python
from_file(vocab, merges, **kwargs)
```

**Parameters:**

vocab (`str`) : The path to a `vocab.json` file 

merges (`str`) : The path to a `merges.txt` file

**Returns:** [BPE](/docs/tokenizers/v0.23.2/en/api/models#tokenizers.models.BPE)

An instance of BPE loaded from these files

Instantiate a BPE model from the given files.

This method is roughly equivalent to doing:

```python
vocab, merges = BPE.read_file(vocab_filename, merges_filename)
bpe = BPE(vocab, merges)
```

If you don't need to keep the `vocab, merges` values lying around,
this method is more optimized than manually calling
`read_file()` to initialize a [BPE](/docs/tokenizers/v0.23.2/en/api/models#tokenizers.models.BPE)

#### read_file[[tokenizers.models.BPE.read_file]]

```python
read_file(vocab, merges)
```

**Parameters:**

vocab (`str`) : The path to a `vocab.json` file 

merges (`str`) : The path to a `merges.txt` file

**Returns:** A `Tuple` with the vocab and the merges

The vocabulary and merges loaded into memory

Read a `vocab.json` and a `merges.txt` files

This method provides a way to read and parse the content of these files,
returning the relevant data structures. If you want to instantiate some BPE models
from memory, this method gives you the expected input from the standard files.

## Model[[tokenizers.models.Model]]

#### tokenizers.models.Model[[tokenizers.models.Model]]

```python
tokenizers.models.Model()
```

Base class for all models

The model represents the actual tokenization algorithm. This is the part that
will contain and manage the learned vocabulary.

This class cannot be constructed directly. Please use one of the concrete models.

#### get_trainer[[tokenizers.models.Model.get_trainer]]

```python
get_trainer()
```

**Returns:** `Trainer`

The Trainer used to train this model

Get the associated `Trainer`

Retrieve the `Trainer` associated to this
[Model](/docs/tokenizers/v0.23.2/en/api/models#tokenizers.models.Model).

#### id_to_token[[tokenizers.models.Model.id_to_token]]

```python
id_to_token(id)
```

**Parameters:**

id (`int`) : An ID to convert to a token

**Returns:** `str`

The token associated to the ID

Get the token associated to an ID

#### save[[tokenizers.models.Model.save]]

```python
save(folder, prefix)
```

**Parameters:**

folder (`str`) : The path to the target folder in which to save the various files 

prefix (`str`, *optional*) : An optional prefix, used to prefix each file name

**Returns:** `List[str]`

The list of saved files

Save the current model

Save the current model in the given folder, using the given prefix for the various
files that will get created.
Any file with the same name that already exists in this folder will be overwritten.

#### token_to_id[[tokenizers.models.Model.token_to_id]]

```python
token_to_id(tokens)
```

**Parameters:**

token (`str`) : A token to convert to an ID

**Returns:** `int`

The ID associated to the token

Get the ID associated to a token

#### tokenize[[tokenizers.models.Model.tokenize]]

```python
tokenize(sequence)
```

**Parameters:**

sequence (`str`) : A sequence to tokenize

**Returns:** A `List` of `Token`

The generated tokens

Tokenize a sequence

## Unigram[[tokenizers.models.Unigram]]

#### tokenizers.models.Unigram[[tokenizers.models.Unigram]]

```python
tokenizers.models.Unigram(vocab = None, unk_id = None, byte_fallback = None, alpha = None, nbest_size = None)
```

**Parameters:**

vocab (`List[Tuple[str, float]]`, *optional*) : A list of vocabulary items and their log-probability scores, e.g. `[("am", -0.2442), ...]`. If not provided, an empty model is created. 

unk_id (`int`, *optional*) : The index of the unknown token in the vocabulary list. 

byte_fallback (`bool`, *optional*, defaults to `False`) : Whether to use SentencePiece byte fallback for characters not in the vocabulary. 

alpha (`float`, *optional*) : A float between 0 and 1 that represents the smoothing parameter (temperature) to use. 

nbest_size (`int`, *optional*) : An integer greater than 0 that represents the maximum number of best paths to consider. If not set, it samples from the full lattice (i.e. all valid subword segmentations).

An implementation of the Unigram algorithm

The Unigram algorithm is a subword tokenization algorithm based on unigram language
models, as used in SentencePiece. It learns a vocabulary by starting with a large
initial vocabulary and iteratively pruning it using the EM algorithm.

Example:

```python
>>> from tokenizers.models import Unigram
>>> # Build an empty model (to be trained)
>>> model = Unigram()
>>> # Build from a vocabulary list
>>> vocab = [("&amp;lt;unk>", 0.0), ("hello", -1.0), ("world", -1.5)]
>>> model = Unigram(vocab=vocab, unk_id=0)
```

## WordLevel[[tokenizers.models.WordLevel]]

#### tokenizers.models.WordLevel[[tokenizers.models.WordLevel]]

```python
tokenizers.models.WordLevel(vocab = None, unk_token = None)
```

**Parameters:**

vocab (`str`, *optional*) : A dictionary of string keys and their ids `&amp;lcub;"am": 0,...}` 

unk_token (`str`, *optional*) : The unknown token to be used by the model.

An implementation of the WordLevel algorithm

Most simple tokenizer model based on mapping tokens to their corresponding id.

Example:

```python
>>> from tokenizers.models import WordLevel
>>> # Build from a vocabulary dictionary
>>> vocab = &amp;lcub;"hello": 0, "world": 1, "&amp;lt;unk>": 2}
>>> model = WordLevel(vocab=vocab, unk_token="REDACTED")
>>> # Load from file
>>> model = WordLevel.from_file("vocab.json", unk_token="REDACTED")
```

#### from_file[[tokenizers.models.WordLevel.from_file]]

```python
from_file(vocab, unk_token = None)
```

**Parameters:**

vocab (`str`) : The path to a `vocab.json` file

**Returns:** [WordLevel](/docs/tokenizers/v0.23.2/en/api/models#tokenizers.models.WordLevel)

An instance of WordLevel loaded from file

Instantiate a WordLevel model from the given file

This method is roughly equivalent to doing:

```python
vocab = WordLevel.read_file(vocab_filename)
wordlevel = WordLevel(vocab)
```

If you don't need to keep the `vocab` values lying around, this method is
more optimized than manually calling `read_file()` to
initialize a [WordLevel](/docs/tokenizers/v0.23.2/en/api/models#tokenizers.models.WordLevel)

#### read_file[[tokenizers.models.WordLevel.read_file]]

```python
read_file(vocab)
```

**Parameters:**

vocab (`str`) : The path to a `vocab.json` file

**Returns:** `Dict[str, int]`

The vocabulary as a `dict`

Read a `vocab.json`

This method provides a way to read and parse the content of a vocabulary file,
returning the relevant data structures. If you want to instantiate some WordLevel models
from memory, this method gives you the expected input from the standard files.

## WordPiece[[tokenizers.models.WordPiece]]

#### tokenizers.models.WordPiece[[tokenizers.models.WordPiece]]

```python
tokenizers.models.WordPiece(vocab = None, unk_token = '[UNK]', max_input_chars_per_word = 100, continuing_subword_prefix = '##')
```

**Parameters:**

vocab (`Dict[str, int]`, *optional*) : A dictionary of string keys and their ids `&amp;lcub;"am": 0,...}` 

unk_token (`str`, *optional*) : The unknown token to be used by the model. 

max_input_chars_per_word (`int`, *optional*) : The maximum number of characters to authorize in a single word.

An implementation of the WordPiece algorithm

Example:

```python
>>> from tokenizers.models import WordPiece
>>> # Build an empty model (to be trained)
>>> model = WordPiece(unk_token="[UNK]")
>>> # Load from a vocabulary file
>>> model = WordPiece.from_file("vocab.txt")
```

#### from_file[[tokenizers.models.WordPiece.from_file]]

```python
from_file(vocab, **kwargs)
```

**Parameters:**

vocab (`str`) : The path to a `vocab.txt` file

**Returns:** [WordPiece](/docs/tokenizers/v0.23.2/en/api/models#tokenizers.models.WordPiece)

An instance of WordPiece loaded from file

Instantiate a WordPiece model from the given file

This method is roughly equivalent to doing:

```python
vocab = WordPiece.read_file(vocab_filename)
wordpiece = WordPiece(vocab)
```

If you don't need to keep the `vocab` values lying around, this method is
more optimized than manually calling `read_file()` to
initialize a [WordPiece](/docs/tokenizers/v0.23.2/en/api/models#tokenizers.models.WordPiece)

#### read_file[[tokenizers.models.WordPiece.read_file]]

```python
read_file(vocab)
```

**Parameters:**

vocab (`str`) : The path to a `vocab.txt` file

**Returns:** `Dict[str, int]`

The vocabulary as a `dict`

Read a `vocab.txt` file

This method provides a way to read and parse the content of a standard *vocab.txt*
file as used by the WordPiece Model, returning the relevant data structures. If you
want to instantiate some WordPiece models from memory, this method gives you the
expected input from the standard files.

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Encode Inputs
https://huggingface.co/docs/tokenizers/v0.23.2/api/encode-inputs.md
