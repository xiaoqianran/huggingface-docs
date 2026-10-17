# PP-DocLayoutV4

## Overview

**PP-DocLayoutV4** is an end-to-end document layout analysis model that predicts, in a single forward pass, what each
layout element is, the quadrilateral that encloses it, and the logits the reading order is decoded from.

## Model Architecture

PP-DocLayoutV4 keeps the RT-DETR skeleton of its predecessor — an HGNetV2 backbone, an AIFI hybrid encoder and a
deformable decoder — and changes three things:

- **Quadrilaterals instead of masks.** [PPDocLayoutV3ForObjectDetection](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v3#transformers.PPDocLayoutV3ForObjectDetection) predicted a segmentation mask per query and
  fitted a polygon to it. PP-DocLayoutV4 drops the mask branch entirely and directly regresses a four point quad,
  encoded as `[center_x, center_y, dx1, dy1, ..., dx4, dy4]` in sigmoid space. This handles skewed and perspective
  distorted documents without paying for a mask head, and the stride 4 backbone feature is no longer needed.
- **Untied per-layer heads.** Every decoder layer owns its own box and classification head rather than sharing a
  single head with the encoder.
- **Two reading order heads plus a fusion step.** Alongside the relative order head from V3 (antisymmetric: "is `i`
  read before `j`?") there is a successor head (ROOR: "does `j` directly follow `i`?"). `PPDocLayoutV4S2RFusion`
  folds a damped transitive closure of the successor matrix back into the relative order logits.

The reading order is not decoded inside the model. `PPDocLayoutV4ForObjectDetection` returns the raw
`relative_order_logits` and `successor_order_logits`, and
[post_process_object_detection()](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4ImageProcessor.post_process_object_detection) turns them into ranks by thresholding the successor
matrix into a directed acyclic graph (DAG), breaking cycles, sorting each connected component topologically, and
ordering the components by their relative order votes.

## Usage

Results returned by [post_process_object_detection()](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4ImageProcessor.post_process_object_detection) are already sorted by reading
order. Each result carries `polygon_points` — a `(num_boxes, 4, 2)` tensor of the regressed corners in top-left,
top-right, bottom-right, bottom-left order — plus `boxes`, the axis aligned rectangle enclosing each quad, and
`order_seq`, the 0-based rank of each box. The [AutoModel](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoModel) example below prints `order_seq` directly, the others
print the 1-based position in the sorted list, which is `order_seq + 1` unless one query was kept under several
labels.

[Pipeline](/docs/transformers/v5.20.1/en/main_classes/pipelines#transformers.Pipeline) preserves the reading order in the order of the returned list, but flattens each detection to a score, a
label and an integer bounding box. Use the [AutoModel](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoModel) path below to get the quadrilaterals and `order_seq`.

```python
from transformers import pipeline
from transformers.image_utils import load_image

image = load_image("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg")
layout_detector = pipeline("object-detection", model="PaddlePaddle/PP-DocLayoutV4_safetensors")
results = layout_detector(image)
for idx, res in enumerate(results):
    print(f"Order {idx + 1}: {res}")
```

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import load_image

model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = load_image("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg")
inputs = image_processor(images=image, return_tensors="pt").to(device=model.device, dtype=model.dtype)

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])
for result in results:
    for score, label_id, box, quad, rank in zip(
        result["scores"], result["labels"], result["boxes"], result["polygon_points"], result["order_seq"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {rank.item()}: {model.config.id2label[label_id.item()]}, score: {score.item():.2f}, box: {box}")
        print(f"         corners: {[[round(x, 2) for x in point] for point in quad.tolist()]}")
```

### Batched inference

Pass a list of images and one target size per image. The reading order is decoded per image.

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import load_image

model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = load_image("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg")
inputs = image_processor(images=[image, image], return_tensors="pt").to(device=model.device, dtype=model.dtype)

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]] * 2)
for result in results:
    print("result:")
    for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {idx + 1}: {model.config.id2label[label_id.item()]}, score: {score.item():.2f}, box: {box}")
```

## PPDocLayoutV4ForObjectDetection[[transformers.PPDocLayoutV4ForObjectDetection]]

#### transformers.PPDocLayoutV4ForObjectDetection[[transformers.PPDocLayoutV4ForObjectDetection]]

```python
transformers.PPDocLayoutV4ForObjectDetection(config: PPDocLayoutV4Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/pp_doclayout_v4/modeling_pp_doclayout_v4.py#L1650)

**Parameters:**

config ([PPDocLayoutV4Config](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

PP-DocLayoutV4 Model (consisting of a backbone and encoder-decoder) outputting quadrilaterals, class logits and
reading order logits, for tasks such as document layout analysis.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.PPDocLayoutV4ForObjectDetection.forward]]

```python
forward(pixel_values: FloatTensor, pixel_mask: typing.Optional[torch.LongTensor] = None, encoder_outputs: typing.Optional[torch.FloatTensor] = None, labels: list[dict] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/pp_doclayout_v4/modeling_pp_doclayout_v4.py#L1668)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) : Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:  - 1 for pixels that are real (i.e. **not masked**), - 0 for pixels that are padding (i.e. **masked**).  [What are attention masks?](../glossary#attention-mask)

encoder_outputs (`torch.FloatTensor`, *optional*) : Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

labels (`list[Dict]` of len `(batch_size,)`, *optional*) : Not supported: PP-DocLayoutV4 is inference only in Transformers.

**Returns:** `PPDocLayoutV4ForObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `PPDocLayoutV4ForObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [PPDocLayoutV4ForObjectDetection](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4ForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`) -- Classification logits (without no-object) for all queries.
- **pred_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`) -- Normalized quads for all queries, encoded as `[center_x, center_y, dx1, dy1, ..., dx4, dy4]` where the corner
  offsets are shifted by `+0.5`. Use
  [post_process_object_detection()](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4ImageProcessor.post_process_object_detection) to retrieve the unnormalized corners and their
  enclosing boxes.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.
- **intermediate_hidden_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) -- Stacked intermediate hidden states (output of each layer of the decoder).
- **intermediate_reference_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_coords)`) -- Stacked intermediate reference points (refined quads of each layer of the decoder).
- **decoder_hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **init_reference_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`) -- Initial quad reference points sent through the Transformer decoder.
- **enc_topk_logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`) -- Class logits of the encoder proposals that were selected as decoder queries.
- **enc_topk_bboxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`) -- Quads of the encoder proposals that were selected as decoder queries.
- **enc_outputs_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Class logits of every encoder proposal.
- **enc_outputs_coord_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_coords)`) -- Quad logits of every encoder proposal.
- **denoising_meta_values** (`dict`, *optional*) -- Extra dictionary for the denoising related values.
- **relative_order_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`) -- Pairwise relative reading order logits, after the optional S2R fusion. A positive `relative_order_logits[i, j]`
  means query `i` is read before query `j`.
- **successor_order_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`) -- Pairwise direct successor (ROOR) logits. A positive `successor_order_logits[i, j]` means query `j` directly
  follows query `i`.

Examples:

```python
>>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
>>> from PIL import Image
>>> import httpx
>>> from io import BytesIO

>>> url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
>>> with httpx.stream("GET", url) as response:
...     image = Image.open(BytesIO(response.read()))

>>> model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
>>> image_processor = AutoImageProcessor.from_pretrained(model_path)
>>> model = AutoModelForObjectDetection.from_pretrained(model_path)

>>> inputs = image_processor(images=[image], return_tensors="pt")
>>> outputs = model(**inputs)

>>> # results are already sorted by reading order
>>> results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])
>>> for result in results:
...     for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"Order {idx + 1}: {model.config.id2label[label_id.item()]}: {score.item():.2f} {box}")
Order 1: text: 0.99 [336.07, 182.04, 894.17, 652.62]
Order 2: paragraph_title: 0.98 [336.45, 681.88, 868.78, 796.91]
Order 3: text: 0.99 [334.01, 840.84, 889.11, 1452.29]
Order 4: text: 0.99 [920.65, 183.62, 1476.75, 462.75]
Order 5: text: 0.99 [919.55, 482.55, 1479.85, 763.41]
Order 6: text: 0.98 [919.34, 844.48, 1481.28, 1219.44]
Order 7: text: 0.98 [920.84, 1238.61, 1467.66, 1374.54]
Order 8: text: 0.90 [334.24, 1612.85, 1478.79, 1730.92]
Order 9: text: 0.95 [334.25, 1755.98, 1467.34, 1845.6]
Order 10: text: 0.65 [337.14, 1909.47, 659.78, 1938.2]
Order 11: footnote: 0.78 [338.73, 2114.87, 1448.16, 2172.06]
Order 12: number: 0.98 [106.08, 2257.42, 134.76, 2281.3]
Order 13: footer: 0.93 [339.27, 2255.7, 984.16, 2282.81]
```

## PPDocLayoutV4Model[[transformers.PPDocLayoutV4Model]]

#### transformers.PPDocLayoutV4Model[[transformers.PPDocLayoutV4Model]]

```python
transformers.PPDocLayoutV4Model(config: PPDocLayoutV4Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/pp_doclayout_v4/modeling_pp_doclayout_v4.py#L1328)

**Parameters:**

config ([PPDocLayoutV4Config](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

PP-DocLayoutV4 Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.PPDocLayoutV4Model.forward]]

```python
forward(pixel_values: FloatTensor, pixel_mask: typing.Optional[torch.LongTensor] = None, encoder_outputs: typing.Optional[torch.FloatTensor] = None, labels: list[dict] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/pp_doclayout_v4/modeling_pp_doclayout_v4.py#L1466)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) : Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:  - 1 for pixels that are real (i.e. **not masked**), - 0 for pixels that are padding (i.e. **masked**).  [What are attention masks?](../glossary#attention-mask)

encoder_outputs (`torch.FloatTensor`, *optional*) : Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

labels (`list[Dict]` of len `(batch_size,)`, *optional*) : Not supported: PP-DocLayoutV4 is inference only in Transformers.

**Returns:** `PPDocLayoutV4ModelOutput` or `tuple(torch.FloatTensor)`

A `PPDocLayoutV4ModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [PPDocLayoutV4Model](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.
- **intermediate_hidden_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) -- Stacked intermediate hidden states (output of each layer of the decoder).
- **intermediate_reference_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_coords)`) -- Stacked intermediate reference points (refined quads of each layer of the decoder).
- **decoder_hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **init_reference_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`) -- Initial quad reference points sent through the Transformer decoder.
- **enc_topk_logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`) -- Class logits of the encoder proposals that were selected as decoder queries.
- **enc_topk_bboxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_coords)`) -- Quads of the encoder proposals that were selected as decoder queries.
- **enc_outputs_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Class logits of every encoder proposal.
- **enc_outputs_coord_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_coords)`) -- Quad logits of every encoder proposal.
- **denoising_meta_values** (`dict`, *optional*) -- Extra dictionary for the denoising related values.
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`) -- Classification logits of the last decoder layer.
- **relative_order_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`) -- Pairwise relative reading order logits, after the optional S2R fusion.
- **successor_order_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_queries, config.num_queries)`) -- Pairwise direct successor (ROOR) logits.

## PPDocLayoutV4Config[[transformers.PPDocLayoutV4Config]]

#### transformers.PPDocLayoutV4Config[[transformers.PPDocLayoutV4Config]]

```python
transformers.PPDocLayoutV4Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, is_encoder_decoder: bool = True, initializer_range: float = 0.01, initializer_bias_prior_prob: float | None = None, layer_norm_eps: float = 1e-05, batch_norm_eps: float = 1e-05, tie_word_embeddings: bool = True, backbone_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, freeze_backbone_batch_norms: bool = True, encoder_hidden_dim: int = 256, encoder_in_channels: list[int] | tuple[int, ...] = (512, 1024, 2048), feat_strides: list[int] | tuple[int, ...] = (8, 16, 32), encoder_layers: int = 1, encoder_ffn_dim: int = 1024, encoder_attention_heads: int = 8, dropout: float | int = 0.0, activation_dropout: float | int = 0.0, encode_proj_layers: list[int] | tuple[int, ...] = (2,), positional_encoding_temperature: int = 10000, encoder_activation_function: str = 'gelu', activation_function: str = 'silu', eval_size: list[int] | tuple[int, int] | None = None, normalize_before: bool = False, hidden_expansion: float = 1.0, num_queries: int = 300, decoder_in_channels: list[int] | tuple[int, ...] = (256, 256, 256), decoder_ffn_dim: int = 1024, num_feature_levels: int = 3, decoder_n_points: int = 4, decoder_layers: int = 6, decoder_attention_heads: int = 8, decoder_activation_function: str = 'relu', attention_dropout: float | int = 0.0, num_denoising: int = 100, anchor_image_size: list[int] | tuple[int, int] | None = None, disable_custom_kernels: bool = True, global_pointer_head_size: int = 64, gp_dropout_value: float | int = 0.1, hidden_size: int = 256, num_coords: int = 10, s2r_steps: int = 3, s2r_damping: float = 0.5, s2r_closure_weight_init: float = 0.0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/pp_doclayout_v4/configuration_pp_doclayout_v4.py#L30)

**Parameters:**

is_encoder_decoder (`bool`, *optional*, defaults to `True`) : Whether the model is used as an encoder/decoder or not.

initializer_range (`float`, *optional*, defaults to `0.01`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_bias_prior_prob (`float`, *optional*) : The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`. If `None`, it defaults to `1 / (num_labels + 1)`.

layer_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

batch_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the batch normalization layers.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

backbone_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The configuration of the backbone model.

freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`) : Whether to freeze the batch normalization layers in the backbone.

encoder_hidden_dim (`int`, *optional*, defaults to `256`) : Dimension of the hidden representations.

encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`) : Multi level features input for encoder.

feat_strides (`list[int]`, *optional*, defaults to `[8, 16, 32]`) : Strides used in each feature map.

encoder_layers (`int`, *optional*, defaults to `1`) : Number of hidden layers in the Transformer encoder. Will use the same value as `num_layers` if not set.

encoder_ffn_dim (`int`, *optional*, defaults to `1024`) : Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.

encoder_attention_heads (`int`, *optional*, defaults to `8`) : Number of attention heads for each attention layer in the Transformer encoder.

dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The ratio for all dropout layers.

activation_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for activations inside the fully connected layer.

encode_proj_layers (`list[int]`, *optional*, defaults to `[2]`) : Indexes of the projected layers to be used in the encoder.

positional_encoding_temperature (`int`, *optional*, defaults to 10000) : The temperature parameter used to create the positional encodings.

encoder_activation_function (`str`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

activation_function (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

eval_size (`tuple[int, int]`, *optional*) : Height and width used to computes the effective height and width of the position embeddings after taking into account the stride. Must be left as `None` to reproduce the reference implementation, which computes the position embeddings dynamically.

normalize_before (`bool`, *optional*, defaults to `False`) : Determine whether to apply layer normalization in the transformer encoder layer before self-attention and feed-forward modules.

hidden_expansion (`float`, *optional*, defaults to 1.0) : Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.

num_queries (`int`, *optional*, defaults to 300) : Number of object queries.

decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`) : Multi level features dimension for decoder

decoder_ffn_dim (`int`, *optional*, defaults to 1024) : Dimension of the "intermediate" (often named feed-forward) layer in decoder.

num_feature_levels (`int`, *optional*, defaults to 3) : The number of input feature levels.

decoder_n_points (`int`, *optional*, defaults to 4) : The number of sampled keys in each feature level for each attention head in the decoder.

decoder_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.

decoder_attention_heads (`int`, *optional*, defaults to `8`) : Number of attention heads for each attention layer in the Transformer decoder.

decoder_activation_function (`str`, *optional*, defaults to `"relu"`) : The non-linear activation function (function or string) in the decoder. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

num_denoising (`int`, *optional*, defaults to 100) : The total number of denoising tasks or queries to be used for contrastive denoising.

anchor_image_size (`tuple[int, int]`, *optional*) : Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.

disable_custom_kernels (`bool`, *optional*, defaults to `True`) : Whether to disable custom kernels.

global_pointer_head_size (`int`, *optional*, defaults to 64) : The size of the global pointer head.

gp_dropout_value (`float`, *optional*, defaults to 0.1) : The dropout probability in the global pointer head.

hidden_size (`int`, *optional*, defaults to 256) : Dimension of the decoder layers, excluding the hybrid encoder. Also readable as `d_model`, the name used by the RT-DETR lineage this model descends from.

num_coords (`int`, *optional*, defaults to 10) : Size of the box parameterization predicted by the bbox heads. PP-DocLayoutV4 regresses a four point quadrilateral encoded as `[center_x, center_y, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]` in sigmoid space, where the corner offsets are shifted by `+0.5`. Only `10` is supported.

s2r_steps (`int`, *optional*, defaults to 3) : Number of propagation steps used to approximate the transitive closure of the successor matrix.

s2r_damping (`float`, *optional*, defaults to 0.5) : Damping factor applied to every additional propagation step of the transitive closure.

s2r_closure_weight_init (`float`, *optional*, defaults to 0.0) : Initial value of the learnable gate that weights the closure term. Defaults to `0.0` so that an untrained fusion module is numerically identical to using the relative order logits alone.

This is the configuration class to store the configuration of a PPDocLayoutV4Model. It is used to instantiate a Pp Doclayout V4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [PaddlePaddle/PP-DocLayoutV4_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV4_safetensors)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import PPDocLayoutV4Config, PPDocLayoutV4ForObjectDetection

>>> # Initializing a PP-DocLayoutV4 configuration
>>> configuration = PPDocLayoutV4Config()

>>> # Initializing a model (with random weights) from the configuration
>>> model = PPDocLayoutV4ForObjectDetection(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## PPDocLayoutV4ImageProcessor[[transformers.PPDocLayoutV4ImageProcessor]]

#### transformers.PPDocLayoutV4ImageProcessor[[transformers.PPDocLayoutV4ImageProcessor]]

```python
transformers.PPDocLayoutV4ImageProcessor(**kwargs: *<class 'transformers.processing_utils.ImagesKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/pp_doclayout_v4/image_processing_pp_doclayout_v4.py#L35)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 800, 'width': 800}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `3`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0, 0, 0]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[1, 1, 1]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models. Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models. Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models. Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models. Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors. --

Constructs a PPDocLayoutV4ImageProcessor image processor.

disable_grouping (`bool`, *kwargs*, *optional*):
Whether to disable grouping of images by size to process them individually and not in batches.
If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
image_seq_length (`int`, *kwargs*, *optional*):
The number of image tokens to be used for each image in the input.
Added for backward compatibility but this should be set as a processor attribute in future models.

#### preprocess[[transformers.PPDocLayoutV4ImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], *args, image_like_kwargs: dict[str, typing.Any] | None = None, **kwargs: *<class 'transformers.processing_utils.ImagesKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/image_processing_utils.py#L382)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

image_like_kwargs (`dict[str, Any]`, *optional*) : Developer flag for additional image like inputs that will also be preprocessed. Only use this if passing the inputs as kwarg doesn't work. For example, `preprocess(images, masks=masks)` is the preferred option but results in argument priority issues for some models. In those cases `preprocess(images, image_like_inputs={"masks": masks})` can be used instead.

do_convert_rgb (`bool`, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

Preprocess an image or a batch of images.

#### post_process_object_detection[[transformers.PPDocLayoutV4ImageProcessor.post_process_object_detection]]

```python
post_process_object_detection(outputs, threshold: float = 0.5, target_sizes = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/pp_doclayout_v4/image_processing_pp_doclayout_v4.py#L120)

**Parameters:**

outputs (`PPDocLayoutV4ForObjectDetectionOutput`) : Raw outputs of the model.

threshold (`float`, *optional*, defaults to 0.5) : Score threshold to keep object detection predictions.

target_sizes (`torch.Tensor` or `list[tuple[int, int]]`) : Tensor of shape `(batch_size, 2)` or list of tuples (`(height, width)`) with the target size of each image in the batch.

**Returns:** `list[Dict]`

A list of dictionaries, one per image, each containing the `scores`, `labels`, `boxes`,
`polygon_points` and `order_seq` predicted by the model. Predictions are sorted by reading order, so
`order_seq` is non-decreasing (a query selected under several labels keeps a single rank).

Converts the raw output of [PPDocLayoutV4ForObjectDetection](/docs/transformers/v5.20.1/en/model_doc/pp_doclayout_v4#transformers.PPDocLayoutV4ForObjectDetection) into final quadrilaterals, enclosing boxes in
`(top_left_x, top_left_y, bottom_right_x, bottom_right_y)` format and reading order ranks. Only supports
PyTorch.

### Cohere2 MoE
https://huggingface.co/docs/transformers/v5.20.1/model_doc/cohere2_moe.md
