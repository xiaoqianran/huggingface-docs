# Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods

    

CPT optimizing only specific token embeddings while keeping the rest of the model frozen (image source).

[Context-Aware Prompt Tuning (CPT)](https://huggingface.co/papers/2410.17222) is designed to enhance few-shot classification by refining only context embeddings.
This approach combines ideas from In-Context Learning (ICL), [Prompt Tuning](../package_reference/prompt_tuning) (PT), and adversarial optimization, focusing on making model adaptation both parameter-efficient and effective.
In CPT, only specific context token embeddings are optimized, while the rest of the model remains frozen.
To prevent overfitting and maintain stability, CPT uses controlled perturbations to limit the allowed changes to context embeddings within a defined range.
Additionally, to address the phenomenon of recency bias—where examples near the end of the context tend to be prioritized over earlier ones—CPT applies a decay loss factor.

The abstract from the paper is:

> Large Language Models (LLMs) can perform few-shot learning using either optimization-based approaches or In-Context Learning (ICL). Optimization-based methods often suffer from overfitting, as they require updating a large number of parameters with limited data. In contrast, ICL avoids overfitting but typically underperforms compared to optimization-based methods and is highly sensitive to the selection, order, and format of demonstration examples. To overcome these challenges, we introduce Context-aware Prompt Tuning (CPT), a method inspired by ICL, Prompt Tuning (PT), and adversarial attacks. CPT builds on the ICL strategy of concatenating examples before the input, extending it by incorporating PT-like learning to refine the context embedding through iterative optimization, extracting deeper insights from the training examples. Our approach carefully modifies specific context tokens, considering the unique structure of the examples within the context. In addition to updating the context with PT-like optimization, CPT draws inspiration from adversarial attacks, adjusting the input based on the labels present in the context while preserving the inherent value of the user-provided data. To ensure robustness and stability during optimization, we employ a projected gradient descent algorithm, constraining token embeddings to remain close to their original values and safeguarding the quality of the context. Our method has demonstrated superior accuracy across multiple classification tasks using various LLM models, outperforming existing baselines and effectively addressing the overfitting challenge in few-shot learning.

Take a look at [Example](https://github.com/huggingface/peft/blob/main/examples/cpt_finetuning/README.md) for a step-by-step guide on how to train a model with CPT.

## Working Principle

Context-aware Prompt Tuning is similar to [Prompt Tuning](./prompt_tuning) in that you can take a text prompt of your
choosing but make the tokens of this prompt, and only this prompt, trainable by virtue of having a second token
embedding over the prompt tokens only. Whenever you query a CPT model, your 'soft' prompt will be prepended to the
embedding of your normal input. CPT differs from Prompt Tuning in that your soft prompt can now be contextualized by
marking specific parts NOT to be trained. The CPT-specific token embedding matrix consists of a static embedding
initialized from the base model's embedding matrix and learnable token-specific delta values.

CPT is useful when you have a prompt with instructions alongside examples and ground-truth/labels. Instructions can be
suboptimal and therefore benefit from being optimized during training (like with prompt tuning) but the examples/labels
you included in the prompt are not supposed to change. This can be enforced by explicitly defining where prompting ends
and examples begin. CPT also adds a token loss to the standard next token prediction loss of your model for the
examples in the prompt to make optimal use of the given examples.

CPT uses five different type ids to mark the different sections in the prompt:

* 0: separator: will be ignored completely, useful for begin/end-of-sentence tokens
* 1: input marker: tokens describing the prompt that marks the whole input-side template, including additional info
* 2: input query: tokens describing the user input of an example
* 3: output marker: tokens describing the start of the expected output of an example
* 4: label: tokens belonging to the label of the example

This is very abstract, so here's an example of a CPT prompt and the corresponding type identifiers.
We assume that each word = 1 token for simplicity:

```
prompt_text = "Example 1: Input: 1+1 Answer: 2 Example 2: 1+3 Answer: 4"

token_ids = [15,  2,      11, 9, 1,    9, 11, 3, 11, 8,     9, 4, 2,       4, 9, 11, 3, 6, 8,     9,  10]
#            BOS, Example 1   :  Input :  1   +  1   Answer :  2  Example  2  :  1   +  3  Answer :   4
type_ids  = [0,   1,      1,  1, 1,    1, 2,  2, 2,  3,     3, 4, 5,       5, 5, 6,  6, 6, 7,     7,  8]
                                                                # 1,       1, 1, 2,  2, 2, 3,     3,  4
```

The `token_ids` are your prompt encoded by the base model's tokenizer and correspond to `~CPTConfig.cpt_token_ids`.
`type_ids` are the prompt token type ids, defining the different sections of the prompt and correspond to
`~CPTConfig.cpt_tokens_type_mask`.

The individual ids have different effects on the training process:

* 0: will be completely ignored, doesn't contribute to the loss and these tokens are not trained, useful for tokens
  that are just there for technical reasons and would otherwise uselessly enlarge the parameter space (e.g., EOS tokens)
* 1 and 3: these tokens are considered for formatting and the maximum norm of their embedding delta is controlled by
  `opt_projection_format_epsilon()`. The norm controls the maximum magnitude in which the learned delta
  embeddings for this type can point. Higher norms mean a higher potential change in meaning.
* 2: these tokens are considered for the input of an example and the maximum norm of their embedding delta is controlled
  by `opt_projection_epsilon()`. The norm controls the maximum magnitude in which the learned delta
  embeddings for this type can point. Higher norms mean a higher potential change in meaning.
* 4: The prompt's label tokens are added as extra next-token-prediction targets to the standard loss on your training
  data; every other soft-prompt token is masked out of the loss. The label token delta embeddings are never changed
  to keep the label semantics intact.

You may have noticed that the numbers in `type_ids` for the second example are shifted by 4. CPT has an exponential
weighting for the loss of the examples you provide in the prompt, the last example being the least decayed and the first
example's loss being the most decaying. This gives more weight to examples closer to the actual input query and is
controlled by `opt_loss_decay_factor()` when `opt_weighted_loss_type()` is set to `"decay"`. By
shifting the examples by 4 we can enumerate the different examples so that the loss can give the different decaying
weightings to each example.

Note that `cpt_tokens_type_mask` only describes the soft prompt itself. When training, additionally pass an
`input_type_mask` tensor alongside each batch, assigning type ids to the tokens of your actual input (using ids 0-4 as
above, with padding or end-of-sentence tokens marked as 0). CPT automatically shifts these ids by the number of
examples in your prompt, so you don't need to shift them yourself. This is then the same as adding the input at the end
of your list of examples in the prompt. Keep in mind that the ids are interpreted modulo 4: shifting them by a multiple
of 4 is harmless, but any other offset silently changes which tokens contribute to the loss.  If no `input_type_mask` is
passed, CPT falls back to computing the loss over the whole input sequence, i.e. assuming type 4 for all input tokens.

## Benchmark overview

There is no benchmark for this method yet. Feel free to contribute an experiment
configuration but make sure to first create an issue
[here](https://github.com/huggingface/peft/issues).

# API

## CPTConfig[[peft.CPTConfig]]

#### peft.CPTConfig[[peft.CPTConfig]]

```python
peft.CPTConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None, cpt_token_ids: list[int] | None = None, cpt_mask: list[int] | None = None, cpt_tokens_type_mask: list[int] | None = None, opt_weighted_loss_type: typing.Literal['none', 'decay'] | None = 'none', opt_loss_decay_factor: float | None = 1.0, opt_projection_epsilon: float | None = 0.1, opt_projection_format_epsilon: float | None = 0.1, tokenizer_name_or_path: str | None = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/cpt/config.py#L23)

CPT Configuration class extending PeftConfig for Context-aware Prompt Tuning (CPT).

This class introduces additional parameters required for CPT, such as:
- Token type masks
- Prompt tuning initialization
- Loss weighting
- Projection settings

For more details, see the paper: https://huggingface.co/papers/2410.17222

## CPTEmbedding[[peft.CPTEmbedding]]

#### peft.CPTEmbedding[[peft.CPTEmbedding]]

```python
peft.CPTEmbedding(config, word_embeddings)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/cpt/model.py#L24)

CPTEmbedding is a custom embedding layer designed for Context-aware Prompt Tuning (CPT) in PEFT. It initializes
embeddings, applies prompt-specific projections, and computes loss using label masks.

#### calculate_loss[[peft.CPTEmbedding.calculate_loss]]

```python
calculate_loss(base_model_output, labels, cpt_type_mask, config)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/cpt/model.py#L144)

**Parameters:**

base_model_output (ModelOutput) : Output from the base model containing logits.

labels (torch.Tensor) : Ground-truth labels for the input tokens.

cpt_type_mask (torch.Tensor) : Token type mask used for filtering valid loss terms.

config (Namespace) : Configuration object containing loss-related hyperparameters.

**Returns:** `ModelOutput`

The base model output with computed loss.

Computes the loss for CPT models with optional exponential decay.

Instead of computing a uniform average cross-entropy loss across the entire sequence, this function isolates
label tokens for each distinct demonstration using the modulo-4 namespace encoded in `cpt_type_mask`. With
`config.opt_weighted_loss_type="decay"`, it applies exponential decay weighting
(`config.opt_loss_decay_factor`) across demonstrations, prioritizing later examples (and the final query
prediction) over earlier demonstrations.

Demonstration Grouping Logic:
* Tokens where `(cpt_type_mask > 0) & (cpt_type_mask % 4 == 0)` represent ground-truth labels.
* The demonstration index `k` is the rank of its label type ID among the distinct label IDs present after
  the next-token shift, in ascending order (0-indexed).
* Cross-entropy loss is computed for each valid label token.
* Given `K` distinct label IDs and decay factor `gamma` (`opt_loss_decay_factor`), the weight for
  demonstration `k` (where `0 <= k < K`) is given by:
  `weight_k = gamma ** (K - 1 - k)`
  The final loss is the sum of weighted token losses divided by the number of valid labels.

Example:
Suppose a sequence contains 2 prompt demonstrations followed by 1 user query (K = 3), each with one valid
label after the next-token shift:
- Demo 1 (Prefix Demo 1): Label tokens have type 4 -> Cross-entropy L_0 = 2.0
- Demo 2 (Prefix Demo 2): Label tokens have type 8 -> Cross-entropy L_1 = 1.6
- Demo 3 (Target Query) : Label tokens have type 12 -> Cross-entropy L_2 = 1.0

With `opt_weighted_loss_type="decay"` and `opt_loss_decay_factor = 0.5`:
- Demo 1 weight: 0.5 ** (3 - 1 - 0) = 0.5 ** 2 = 0.25
- Demo 2 weight: 0.5 ** (3 - 1 - 1) = 0.5 ** 1 = 0.50
- Demo 3 weight: 0.5 ** (3 - 1 - 2) = 0.5 ** 0 = 1.00

Weighted Loss:
loss = (0.25 * 2.0 + 0.50 * 1.6 + 1.00 * 1.0) / 3
= (0.5 + 0.8 + 1.0) / 3 = 2.3 / 3 = 0.7666...

#### forward[[peft.CPTEmbedding.forward]]

```python
forward(indices)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/cpt/model.py#L66)

**Parameters:**

indices (torch.Tensor) : Indices of the tokens to be embedded.

**Returns:** `torch.Tensor`

Sum of prompt embeddings and delta embeddings.

Computes the prompt embeddings and applies delta adjustments.

#### get_projection[[peft.CPTEmbedding.get_projection]]

```python
get_projection()
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/cpt/model.py#L126)

Applies epsilon-based projection to the delta embeddings to control their norm.

#### set_updated_tokens[[peft.CPTEmbedding.set_updated_tokens]]

```python
set_updated_tokens()
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/cpt/model.py#L87)

Sets up a backward hook to selectively update token gradients based on the CPT token type mask.

### Weight-Decomposed Low-Rank Adaptation (DoRA)
https://huggingface.co/docs/peft/v0.22.0/package_reference/lora_variant_dora.md
