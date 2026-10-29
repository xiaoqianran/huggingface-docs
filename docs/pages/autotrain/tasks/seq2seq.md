# Seq2Seq

Seq2Seq is a task that involves converting a sequence of words into another sequence of words.
It is used in machine translation, text summarization, and question answering.

## Data Format

You can have the dataset as a CSV file:

```csv
text,target
"this movie is great","dieser Film ist großartig"
"this movie is bad","dieser Film ist schlecht"
.
.
.
```

Or as a JSONL file:

```json
{"text": "this movie is great", "target": "dieser Film ist großartig"}
{"text": "this movie is bad", "target": "dieser Film ist schlecht"}
.
.
.
```

## Columns

Your CSV/JSONL dataset must have two columns: `text` and `target`.

## Parameters[[autotrain.trainers.seq2seq.params.Seq2SeqParams]]

#### autotrain.trainers.seq2seq.params.Seq2SeqParams[[autotrain.trainers.seq2seq.params.Seq2SeqParams]]

```python
autotrain.trainers.seq2seq.params.Seq2SeqParams(data_path: str = None, model: str = 'google/flan-t5-base', username: typing.Optional[str] = None, seed: int = 42, train_split: str = 'train', valid_split: typing.Optional[str] = None, project_name: str = 'project-name', token: typing.Optional[str] = None, push_to_hub: bool = False, text_column: str = 'text', target_column: str = 'target', lr: float = 5e-05, epochs: int = 3, max_seq_length: int = 128, max_target_length: int = 128, batch_size: int = 2, warmup_ratio: float = 0.1, gradient_accumulation: int = 1, optimizer: str = 'adamw_torch', scheduler: str = 'linear', weight_decay: float = 0.0, max_grad_norm: float = 1.0, logging_steps: int = -1, eval_strategy: str = 'epoch', auto_find_batch_size: bool = False, mixed_precision: typing.Optional[str] = None, save_total_limit: int = 1, peft: bool = False, quantization: typing.Optional[str] = 'int8', lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05, target_modules: str = 'all-linear', log: str = 'none', early_stopping_patience: int = 5, early_stopping_threshold: float = 0.01)
```

[Source](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/seq2seq/params.py#L8)

**Parameters:**

data_path (str) : Path to the dataset.

model (str) : Name of the model to be used. Default is "google/flan-t5-base".

username (Optional[str]) : Hugging Face Username.

seed (int) : Random seed for reproducibility. Default is 42.

train_split (str) : Name of the training data split. Default is "train".

valid_split (Optional[str]) : Name of the validation data split.

project_name (str) : Name of the project or output directory. Default is "project-name".

token (Optional[str]) : Hub Token for authentication.

push_to_hub (bool) : Whether to push the model to the Hugging Face Hub. Default is False.

text_column (str) : Name of the text column in the dataset. Default is "text".

target_column (str) : Name of the target text column in the dataset. Default is "target".

lr (float) : Learning rate for training. Default is 5e-5.

epochs (int) : Number of training epochs. Default is 3.

max_seq_length (int) : Maximum sequence length for input text. Default is 128.

max_target_length (int) : Maximum sequence length for target text. Default is 128.

batch_size (int) : Training batch size. Default is 2.

warmup_ratio (float) : Proportion of warmup steps. Default is 0.1.

gradient_accumulation (int) : Number of gradient accumulation steps. Default is 1.

optimizer (str) : Optimizer to be used. Default is "adamw_torch".

scheduler (str) : Learning rate scheduler to be used. Default is "linear".

weight_decay (float) : Weight decay for the optimizer. Default is 0.0.

max_grad_norm (float) : Maximum gradient norm for clipping. Default is 1.0.

logging_steps (int) : Number of steps between logging. Default is -1 (disabled).

eval_strategy (str) : Evaluation strategy. Default is "epoch".

auto_find_batch_size (bool) : Whether to automatically find the batch size. Default is False.

mixed_precision (Optional[str]) : Mixed precision training mode (fp16, bf16, or None).

save_total_limit (int) : Maximum number of checkpoints to save. Default is 1.

peft (bool) : Whether to use Parameter-Efficient Fine-Tuning (PEFT). Default is False.

quantization (Optional[str]) : Quantization mode (int4, int8, or None). Default is "int8".

lora_r (int) : LoRA-R parameter for PEFT. Default is 16.

lora_alpha (int) : LoRA-Alpha parameter for PEFT. Default is 32.

lora_dropout (float) : LoRA-Dropout parameter for PEFT. Default is 0.05.

target_modules (str) : Target modules for PEFT. Default is "all-linear".

log (str) : Logging method for experiment tracking. Default is "none".

early_stopping_patience (int) : Patience for early stopping. Default is 5.

early_stopping_threshold (float) : Threshold for early stopping. Default is 0.01.

Seq2SeqParams is a configuration class for sequence-to-sequence training parameters.

### Object Detection
https://huggingface.co/docs/autotrain/main/tasks/object_detection.md
