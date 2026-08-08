# Post-processors

## BertProcessing[[tokenizers.processors.BertProcessing]]

#### tokenizers.processors.BertProcessing[[tokenizers.processors.BertProcessing]]

This post-processor takes care of adding the special tokens needed by
a Bert model:

- a SEP token
- a CLS token

Example:

```python
>>> from tokenizers.processors import BertProcessing
>>> processor = BertProcessing(("[SEP]", 102), ("[CLS]", 101))
>>> processor.process(encoding)
# Encoding with [CLS] at start and [SEP] at end
```

**Parameters:**

sep (`Tuple[str, int]`) : A tuple with the string representation of the SEP token, and its id 

cls (`Tuple[str, int]`) : A tuple with the string representation of the CLS token, and its id

## ByteLevel[[tokenizers.processors.ByteLevel]]

#### tokenizers.processors.ByteLevel[[tokenizers.processors.ByteLevel]]

This post-processor takes care of trimming the offsets.

By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
want the offsets to include these whitespaces, then this PostProcessor must be used.

Example:

```python
>>> from tokenizers.processors import ByteLevel
>>> processor = ByteLevel(trim_offsets=True)
>>> # Offsets will be trimmed to exclude leading whitespace bytes
```

**Parameters:**

trim_offsets (`bool`) : Whether to trim the whitespaces from the produced offsets. 

add_prefix_space (`bool`, *optional*, defaults to `True`) : If `True`, keeps the first token's offset as is. If `False`, increments the start of the first token's offset by 1. Only has an effect if `trim_offsets` is set to `True`.

## RobertaProcessing[[tokenizers.processors.RobertaProcessing]]

#### tokenizers.processors.RobertaProcessing[[tokenizers.processors.RobertaProcessing]]

This post-processor takes care of adding the special tokens needed by
a Roberta model:

- a SEP token
- a CLS token

It also takes care of trimming the offsets.
By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
want the offsets to include these whitespaces, then this PostProcessor should be initialized
with `trim_offsets=True`

Example:

```python
>>> from tokenizers.processors import RobertaProcessing
>>> processor = RobertaProcessing(("&amp;lt;/s>", 2), ("", 0))
>>> processor.process(encoding)
# Encoding with &amp;lt;s> at start and  at end
```

**Parameters:**

sep (`Tuple[str, int]`) : A tuple with the string representation of the SEP token, and its id 

cls (`Tuple[str, int]`) : A tuple with the string representation of the CLS token, and its id 

trim_offsets (`bool`, *optional*, defaults to `True`) : Whether to trim the whitespaces from the produced offsets. 

add_prefix_space (`bool`, *optional*, defaults to `True`) : Whether the add_prefix_space option was enabled during pre-tokenization. This is relevant because it defines the way the offsets are trimmed out.

## TemplateProcessing[[tokenizers.processors.TemplateProcessing]]

#### tokenizers.processors.TemplateProcessing[[tokenizers.processors.TemplateProcessing]]

Provides a way to specify templates in order to add the special tokens to each
input sequence as relevant.

Let's take `BERT` tokenizer as an example. It uses two special tokens, used to
delimitate each sequence. `[CLS]` is always used at the beginning of the first
sequence, and `[SEP]` is added at the end of both the first, and the pair
sequences. The final result looks like this:

- Single sequence: `[CLS] Hello there [SEP]`
- Pair sequences: `[CLS] My name is Anthony [SEP] What is my name? [SEP]`

With the type ids as following:

```python
[CLS]   ...   [SEP]   ...   [SEP]
0      0      0      1      1
```

You can achieve such behavior using a TemplateProcessing:

```python
TemplateProcessing(
    single="[CLS] $0 [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
)
```

In this example, each input sequence is identified using a `$` construct. This identifier
lets us specify each input sequence, and the type_id to use. When nothing is specified,
it uses the default values. Here are the different ways to specify it:

- Specifying the sequence, with default `type_id == 0`: `$A` or `$B`
- Specifying the *type_id* with default `sequence == A`: `$0`, `$1`, `$2`, ...
- Specifying both: `$A:0`, `$B:1`, ...

The same construct is used for special tokens: `&amp;lt;identifier>(:&amp;lt;type_id>)?`.

**Warning**: You must ensure that you are giving the correct tokens/ids as these
will be added to the Encoding without any further check. If the given ids correspond
to something totally different in a *Tokenizer* using this *PostProcessor*, it
might lead to unexpected results.

Types:

Template (`str` or `List`):
- If a `str` is provided, the whitespace is used as delimiter between tokens
- If a `List[str]` is provided, a list of tokens

Tokens (`List[Union[Tuple[int, str], Tuple[str, int], dict]]`):
- A `Tuple` with both a token and its associated ID, in any order
- A `dict` with the following keys:
  - "id": `str` => The special token id, as specified in the Template
  - "ids": `List[int]` => The associated IDs
  - "tokens": `List[str]` => The associated tokens

  The given dict expects the provided `ids` and `tokens` lists to have
  the same length.

**Parameters:**

single (`Template`) : The template used for single sequences 

pair (`Template`) : The template used when both sequences are specified 

special_tokens (`Tokens`) : The list of special tokens used in each sequences

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Models
https://huggingface.co/docs/tokenizers/v0.23.1/api/models.md
