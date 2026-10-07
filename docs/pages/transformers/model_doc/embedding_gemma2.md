# EmbeddingGemma2

## Overview

EmbeddingGemma 2 is a multimodal embedding model from Google built on the [Gemma 4](./gemma4) architecture. It encodes **text, images, audio, and video**—either individually or combined within the same input—into a shared 768-dimensional dense vector space for cross-modal retrieval, semantic similarity, clustering, and classification.

Key features:

- **Unified multimodal vector space.** Vision ([Gemma4VisionModel](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4VisionModel)) and audio ([Gemma4AudioModel](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4AudioModel)) towers project into a bidirectional text encoder ([EmbeddingGemma2TextModel](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2TextModel)) that interleaves full and sliding-window attention with Per-Layer Embeddings (PLE). Any single modality (`text`, `image`, `audio`, `video`) or combination of modalities (`image + text`, `text + audio`, `image + audio`, multiple images, interleaved text and media) maps to a single comparable embedding.
- **Matryoshka Representation Learning (MRL).** [EmbeddingGemma2Model](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Model) outputs token representations projected to `embedding_dim` (`768`) via a linear head. Embeddings can be truncated to a smaller prefix (`512`, `256`, or `128`) and re-normalized with minimal quality loss.
- **Configurable visual & video budgets.** [EmbeddingGemma2Processor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Processor) lets you tune the soft-token budget per image or frame (`max_soft_tokens` in `{70, 140, 280, 560, 1120}`) and video sampling rate (`fps=1`, `max_frames=32`, `overflow_strategy="uniform"`, `add_timestamps=False` by default).
- **Selective modality tower loading.** Unused vision or audio towers can be disabled at load time (`vision_config=None`, `audio_config=None`) to reduce memory footprint for text-only or single-modality deployments.

You can find all the original EmbeddingGemma checkpoints under the [EmbeddingGemma](https://huggingface.co/collections/google/embeddinggemma) collection. The examples below use the `google/embeddinggemma-2` identifier.

## Usage examples

A sentence or multimodal embedding is obtained in two steps: mask-aware mean pooling over the non-padded tokens of `last_hidden_state`, followed by L2 normalization in `float32`. [Sentence Transformers](https://sbert.net) (`>=6.1.0`) performs preprocessing, prompt formatting, mean pooling, and normalization automatically and is the recommended entry point. With [AutoModel](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoModel), use `processor.apply_chat_template` (or `AutoTokenizer` for plain text) and pool `last_hidden_state` directly.

### Text retrieval (`encode_query` / `encode_document`)

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

queries = ["Which planet is known as the Red Planet?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]

# encode_query and encode_document apply the "query" and "document" task prompts
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (1, 768) (2, 768)

# (1, 2) matrix of cosine similarities
print(model.similarity(query_embeddings, document_embeddings))
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-2")

# Prepend the task prompt directly to plain text inputs (see Task prompts below)
sentences = [
    "task: search result | query: Which planet is known as the Red Planet?",
    "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
    "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    # (batch_size, sequence_length, config.text_config.embedding_dim)
    token_embeddings = model(**inputs).last_hidden_state

# Mask-aware mean pooling over non-padded tokens, then L2 normalization in float32
mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
sentence_embeddings = F.normalize(sentence_embeddings.float(), p=2, dim=-1)

query_embedding, document_embeddings = sentence_embeddings[:1], sentence_embeddings[1:]
print(query_embedding @ document_embeddings.T)
```

### Task prompts

The model supports optional task prompts prepended to the input (and included in mean pooling), though prompts are not mandatory and the model also works without them. Because the optimal setup depends on the downstream domain and modality mix, we recommend evaluating both with and without task prompts on your specific task. Sentence Transformers ships the catalog below in `config_sentence_transformers.json`, so pass `prompt_name` (or use `encode_query` / `encode_document`, which map to `query` and `document`); with [AutoModel](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoModel), pass the prompt as a `system` message in `apply_chat_template` (or prepend it to plain text).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

embeddings = model.encode("How to train a neural network", prompt_name="Classification")
```

| `prompt_name` | prompt |
|---|---|
| `query`, `Retrieval-query`, `Retrieval`, `Reranking`, `BitextMining`, `SearchQuery` | `task: search result \| query: ` |
| `document`, `Document`, `Retrieval-document` | `title: none \| text: ` |
| `Classification`, `MultilabelClassification` | `task: classification \| query: ` |
| `Clustering` | `task: clustering \| query: ` |
| `CodeRetrieval`, `InstructionRetrieval` | `task: code retrieval \| query: ` |
| `FactChecking` | `task: fact checking \| query: ` |
| `QuestionAnswering` | `task: question answering \| query: ` |
| `STS`, `SentenceSimilarity`, `PairClassification`, `Summarization` | `task: sentence similarity \| query: ` |

### Matryoshka embeddings

The embedding head is trained with Matryoshka Representation Learning, so a 768-dimensional embedding can be sliced to a shorter prefix (`512`, `256`, or `128`) and re-normalized. This shrinks the index and speeds up retrieval while largely preserving ranking quality. Truncate queries and documents to the *same* dimension, and always normalize after slicing — in Sentence Transformers, `truncate_dim` slices the already-normalized output, so the prefix is no longer unit-length.

```python
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# truncate_dim can also be set once at load time: SentenceTransformer(..., truncate_dim=256)
model = SentenceTransformer("google/embeddinggemma-2")

queries = ["Which planet is known as the Red Planet?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]

query_embeddings = model.encode_query(queries, truncate_dim=256, convert_to_tensor=True)
document_embeddings = model.encode_document(documents, truncate_dim=256, convert_to_tensor=True)

query_embeddings = F.normalize(query_embeddings.float(), p=2, dim=-1)
document_embeddings = F.normalize(document_embeddings.float(), p=2, dim=-1)

# (1, 256) and (2, 256): a 3x smaller index than the full 768 dimensions
print(query_embeddings.shape, document_embeddings.shape)

# the Mars document still ranks first
print(model.similarity(query_embeddings, document_embeddings))
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-2")

sentences = [
    "task: search result | query: Which planet is known as the Red Planet?",
    "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
    "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

# Slice to the first 256 dimensions, then normalize
sentence_embeddings = F.normalize(sentence_embeddings[:, :256].float(), p=2, dim=-1)
print(sentence_embeddings.shape)
# torch.Size([3, 256])

query_embedding, document_embeddings = sentence_embeddings[:1], sentence_embeddings[1:]

# the Mars document still ranks first
print(query_embedding @ document_embeddings.T)
```

## Multimodal embeddings

Text, images, audio, and video are mapped into the same 768-dimensional vector space, enabling **cross-modal retrieval** (e.g. searching images, audio, or videos with a text query) as well as **composed multimodal retrieval** (combining multiple modalities into a single query or document embedding).

In Sentence Transformers, multimodal inputs are passed as dictionaries keyed by `"text"`, `"image"`, `"audio"`, and `"video"` (where each value can be a single item or a list of items). With [AutoModel](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoModel), pass chat-style message lists to `processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")`.

### 1. Single modalities & cross-modal retrieval

Each modality (`text`, `image`, `audio`, `video`) can be embedded on its own and compared directly against any other modality via cosine similarity:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# Embed each modality on its own into the shared (768,) space
candidates = model.encode(
    [
        {"text": "A photo of a cat"},
        {"image": IMAGE},
        {"audio": "path/to/audio.wav"},
        {"video": "path/to/video.mp4"},
    ]
)
print(candidates.shape)
# (4, 768)

# Cross-modal retrieval: rank text, image, audio, and video candidates against a text query
query = model.encode_query(["A fluffy cat outdoors"])
print(model.similarity(query, candidates))
# tensor([[...]]) of shape (1, 4)
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

conversations = [
    [{"role": "user", "content": [{"type": "text", "text": "A photo of a cat"}]}],
    [{"role": "user", "content": [{"type": "image", "url": IMAGE}]}],
    [{"role": "user", "content": [{"type": "audio", "url": "path/to/audio.wav"}]}],
    [{"role": "user", "content": [{"type": "video", "url": "path/to/video.mp4"}]}],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
print(embeddings.shape)
# torch.Size([4, 768])
```

### 2. Composed multimodal embeddings (several modalities in one input)

Multiple modalities—or multiple items of the same modality—can be combined into a **single joint embedding**, including purely non-text combinations such as `image + audio`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")
IMAGE_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
IMAGE_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

composed_embeddings = model.encode(
    [
        # text + image
        {"text": "A photo of a cat", "image": IMAGE_1},
        # text + audio
        {"text": "A song", "audio": "path/to/audio.wav"},
        # image + audio (no text required)
        {"image": IMAGE_1, "audio": "path/to/audio.wav"},
        # multiple images + audio + text in one embedding
        {"image": [IMAGE_1, IMAGE_2], "audio": "path/to/audio.wav", "text": "Compare both cats"},
    ]
)
print(composed_embeddings.shape)
# (4, 768)
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

conversations = [
    # text + image
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A photo of a cat"},
                {"type": "image", "url": IMAGE},
            ],
        }
    ],
    # text + audio
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A song"},
                {"type": "audio", "url": "path/to/audio.wav"},
            ],
        }
    ],
    # image + audio (no text required)
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": IMAGE},
                {"type": "audio", "url": "path/to/audio.wav"},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
print(embeddings.shape)
# torch.Size([3, 768])
```

### 3. Automatic ordering vs. manual placeholders

When no placeholder tokens (`<|image|>`, `<|video|>`, `<|audio|>`) appear in the text, modalities are emitted in the exact order their entries appear in the dictionary or chat message `content` (immediately after any `system` task prompt).

To interleave text and media at specific positions within a sentence, write `<|image|>`, `<|video|>`, or `<|audio|>` directly in the text—automatic placeholder insertion is then disabled for that input, and each placeholder is expanded in-order with the supplied media items:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")
IMAGE_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
IMAGE_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

# 1. Automatic ordering: follows the dictionary's key order
#    -> <bos> [prompt] <image_tokens> <text_tokens> <eos>
image_first = model.encode({"image": IMAGE_1, "text": "A photo of a cat"}, prompt_name="document")

#    -> <bos> [prompt] <text_tokens> <image_tokens> <eos>
text_first = model.encode({"text": "A photo of a cat", "image": IMAGE_1}, prompt_name="document")

# 2. Manual placeholders: interleave media at exact positions in the text
#    No extra placeholders are inserted; counts must match the passed media inputs.
interleaved = model.encode(
    {
        "text": "A jacket similar to <|image|> or <|image|> featured in <|audio|>",
        "image": [IMAGE_1, IMAGE_2],
        "audio": "path/to/audio.wav",
    },
    prompt_name="query",
)
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
IMAGE_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

# 1. Automatic ordering: follows the content list order after any system prompt
image_first_messages = [
    {"role": "system", "content": "title: none | text: "},
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_1},
            {"type": "text", "text": "A photo of a cat"},
        ],
    },
]

# 2. Manual placeholders: interleave media at exact positions in the text
interleaved_messages = [
    {"role": "system", "content": "task: search result | query: "},
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_1},
            {"type": "image", "url": IMAGE_2},
            {"type": "audio", "url": "path/to/audio.wav"},
            {"type": "text", "text": "A jacket similar to <|image|> or <|image|> featured in <|audio|>"},
        ],
    },
]

inputs = processor.apply_chat_template(
    [image_first_messages, interleaved_messages],
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
```

### 4. Heterogeneous (mixed-modality) batching

A single batch can mix plain text, single-modality media, and multi-modality inputs in one forward pass, producing the same embedding per row as encoding each item individually:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

embeddings = model.encode(
    [
        "A photo of a cat",
        {"image": IMAGE},
        {"text": "A photo of a cat", "image": IMAGE},
        {"audio": "path/to/audio.wav"},
    ],
    prompt_name="document",
)
print(embeddings.shape)
# (4, 768)
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

conversations = [
    [
        {"role": "system", "content": "title: none | text: "},
        {"role": "user", "content": [{"type": "text", "text": "A photo of a cat"}]},
    ],
    [
        {"role": "system", "content": "title: none | text: "},
        {"role": "user", "content": [{"type": "image", "url": IMAGE}]},
    ],
    [
        {"role": "system", "content": "title: none | text: "},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A photo of a cat"},
                {"type": "image", "url": IMAGE},
            ],
        },
    ],
    [
        {"role": "system", "content": "title: none | text: "},
        {"role": "user", "content": [{"type": "audio", "url": "path/to/audio.wav"}]},
    ],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
print(embeddings.shape)
# torch.Size([4, 768])
```

### 5. Disabling unused modality towers (memory optimization)

If your workload only embeds a subset of modalities (for example, **text-only** or **text + images** without audio), you can skip instantiating and loading the unused vision or audio towers by setting `vision_config=None` and/or `audio_config=None` on the config. The unused tower weights in the checkpoint are ignored cleanly without warnings, dropping the model from 744M to 439M parameters without the audio tower, or to 271M with neither tower.

Pass these through `config_kwargs` (which reaches `AutoConfig.from_pretrained`), not `model_kwargs`:

```python
from sentence_transformers import SentenceTransformer

# Text-only deployment (skips both vision and audio towers)
text_model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"vision_config": None, "audio_config": None},
)

# Vision + text deployment (skips audio tower)
vision_text_model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"audio_config": None},
)
```

```python
from transformers import AutoModel

# Text-only deployment (skips both vision and audio towers)
text_model = AutoModel.from_pretrained(
    "google/embeddinggemma-2",
    vision_config=None,
    audio_config=None,
    device_map="auto",
)

# Vision + text deployment (skips audio tower)
vision_text_model = AutoModel.from_pretrained(
    "google/embeddinggemma-2",
    audio_config=None,
    device_map="auto",
)
```

## Processor

[EmbeddingGemma2Processor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Processor) bundles the tokenizer, the image processor, the audio feature extractor, and the video processor.

> [!TIP]
> For batched multimodal inputs, we recommend passing per-sample dictionaries through **Sentence Transformers** (`model.encode([{"image": ..., "text": ...}, ...])`). For maximum control over exact modality ordering and interleaving, include `<|image|>`, `<|video|>`, and `<|audio|>` placeholders manually in `text`.

### Controlling visual token budget (`max_soft_tokens`)

The soft-token budget (`max_soft_tokens`, supported values `{70, 140, 280, 560, 1120}`) controls the resolution and number of visual tokens produced per image or video frame (defaulting to `280` per image and `140` per video frame). Lower values reduce sequence length and latency:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

image_embedding = model.encode(
    {"image": IMAGE},
    processing_kwargs={"image": {"max_soft_tokens": 70}},
)
```

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

inputs = processor(images=[IMAGE], max_soft_tokens=70, return_tensors="pt")
print(inputs["input_ids"].shape, inputs["pixel_values"].shape)
# torch.Size([1, 67]) torch.Size([1, 630, 768])
```

### Video frame sampling controls

By default, [EmbeddingGemma2VideoProcessor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2VideoProcessor) samples frames at 1 FPS (`fps=1`), caps each clip at 32 frames (`max_frames=32`, `overflow_strategy="uniform"`), and omits frame timestamps from the prompt (`add_timestamps=False`). Each parameter can be overridden per call:

- `fps` (`int | float | None`): target sampling rate in frames per second. Requires `VideoMetadata` with valid `fps` and `duration` (automatically populated when decoding from a video file/URL). For pre-decoded frame arrays without metadata, FPS sampling is skipped with a warning and `max_frames` is applied directly.
- `max_frames` (`int | None`): maximum number of frames retained per video.
- `overflow_strategy` (`"uniform" | "truncate" | None`): how excess frames above `max_frames` are reduced (`"uniform"` resamples evenly across the clip; `"truncate"` keeps the leading `max_frames` frames).
- `add_timestamps` (`bool`): whether to prefix each frame's soft-token block with its `mm:ss` timestamp. Unlike `fps`, this has no fallback: `add_timestamps=True` on a video whose metadata has no `fps` raises, because a guessed rate would write wrong `mm:ss` labels into the prompt.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

video_embedding = model.encode(
    {"video": "path/to/video.mp4"},
    processing_kwargs={"video": {"add_timestamps": True, "fps": 2, "max_frames": 16, "overflow_strategy": "truncate"}},
)
```

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

inputs = processor(
    videos=["path/to/video.mp4"],
    add_timestamps=True,
    fps=2,
    max_frames=16,
    overflow_strategy="truncate",
    return_tensors="pt",
)
```

### Direct `processor(...)` calls without chat template

While `processor.apply_chat_template` is the primary entry point for multimodal conversations, `processor(...)` can also be called directly. When `text` is omitted (`text=None`), `<|image|>`, `<|video|>`, and `<|audio|>` placeholders are synthesized automatically for each sample in the batch:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
AUDIO = "path/to/audio.wav"  # a path, a URL, or a 1-D float array sampled at 16 kHz

# Text only
inputs = processor(text=["task: search result | query: Which planet is the Red Planet?"], return_tensors="pt")

# Media only (text=None): placeholders are synthesized per sample
inputs = processor(images=[IMAGE], return_tensors="pt")
inputs = processor(videos=["path/to/video.mp4"], return_tensors="pt")
inputs = processor(audio=[AUDIO], return_tensors="pt")

# Nested per-sample lists or combined modalities with text=None
inputs = processor(
    images=[[IMAGE, IMAGE], [IMAGE]],
    audio=[[AUDIO], [AUDIO, AUDIO]],
    return_tensors="pt",
)

# Manual placeholders in text for direct processor calls
inputs = processor(text=["<|image|> a photo of a cat"], images=[[IMAGE]], return_tensors="pt")
print(list(inputs.keys()))
# ['input_ids', 'attention_mask', 'pixel_values', 'image_position_ids']
```

## EmbeddingGemma2TextConfig[[transformers.EmbeddingGemma2TextConfig]]

#### transformers.EmbeddingGemma2TextConfig[[transformers.EmbeddingGemma2TextConfig]]

```python
transformers.EmbeddingGemma2TextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 262144, hidden_size: int = 512, intermediate_size: int = 2048, num_hidden_layers: int = 24, num_attention_heads: int = 4, num_key_value_heads: int = 2, head_dim: int = 256, hidden_activation: str = 'gelu_pytorch_tanh', max_position_embeddings: int = 262144, initializer_range: float = 0.02, rms_norm_eps: float = 1e-06, pad_token_id: int | None = 0, eos_token_id: int | list[int] | None = 1, bos_token_id: int | None = 2, rope_parameters: dict | None = None, attention_bias: bool = False, attention_dropout: int | float | None = 0.0, sliding_window: int = 512, layer_types: list[str] | None = None, hidden_size_per_layer_input: int = 512, embedding_dim: int = 768)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/configuration_embedding_gemma2.py#L37)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `262144`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `512`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `2048`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `24`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `4`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `2`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `256`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

hidden_activation (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `262144`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

pad_token_id (`int`, *optional*, defaults to `0`) : Token id used for padding in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `1`) : Token id used for end-of-stream in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `2`) : Token id used for beginning-of-stream in the vocabulary.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[int, float]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

sliding_window (`int`, *optional*, defaults to 512) : Inclusive radius of the bidirectional sliding window: a `sliding_attention` layer attends to every position with `abs(q_idx - kv_idx) <= sliding_window`. It is a radius and not the one-sided width the causal Gemma models configure, because the mask here is symmetric, so the default is half of the 1024-wide window the reference implementation states.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

hidden_size_per_layer_input (`int`, *optional*, defaults to 512) : Dimensionality of the per-layer (PLE) residual signal. EmbeddingGemma 2 uses *projection-only* PLE: the signal is derived from `inputs_embeds` alone, with no auxiliary token lookup table.

embedding_dim (`int`, *optional*, defaults to 768) : Dimensionality of the sentence embedding produced by `embedding_projection`.

This is the configuration class to store the configuration of a EmbeddingGemma2Model. It is used to instantiate a Embedding Gemma2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## EmbeddingGemma2Config[[transformers.EmbeddingGemma2Config]]

#### transformers.EmbeddingGemma2Config[[transformers.EmbeddingGemma2Config]]

```python
transformers.EmbeddingGemma2Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: transformers.models.embedding_gemma2.configuration_embedding_gemma2.EmbeddingGemma2TextConfig | dict[str, typing.Any] | None = None, vision_config: transformers.configuration_utils.PreTrainedConfig | dict[str, typing.Any] | None = None, audio_config: transformers.configuration_utils.PreTrainedConfig | dict[str, typing.Any] | None = None, boi_token_id: int | None = 255999, eoi_token_id: int | None = 258882, image_token_id: int | None = 258880, video_token_id: int | None = 258884, boa_token_id: int | None = 256000, eoa_token_index: int | None = 258883, audio_token_id: int | None = 258881, initializer_range: float | None = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/configuration_embedding_gemma2.py#L139)

**Parameters:**

text_config (`EmbeddingGemma2TextConfig`, *optional*) : Configuration of the text backbone.

vision_config (`PreTrainedConfig` or `dict`, *optional*) : Configuration of the vision tower. Reused verbatim from Gemma 4; the tower itself is resolved at runtime through `AutoModel`.

audio_config (`PreTrainedConfig` or `dict`, *optional*) : Configuration of the audio tower. Reused verbatim from Gemma 4; the tower itself is resolved at runtime through `AutoModel`.

boi_token_id (`int`, *optional*, defaults to 255999) : The begin-of-image token index to wrap the image prompt.

eoi_token_id (`int`, *optional*, defaults to 258882) : The end-of-image token index to wrap the image prompt.

image_token_id (`int`, *optional*, defaults to `258880`) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*, defaults to `258884`) : The video token index used as a placeholder for input videos.

boa_token_id (`int`, *optional*, defaults to 256000) : The begin-of-audio token index to wrap the audio prompt.

eoa_token_index (`int`, *optional*, defaults to 258883) : The end-of-audio token index to wrap the audio prompt.

audio_token_id (`int`, *optional*, defaults to `258881`) : The audio token index used as a placeholder for input audio.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a EmbeddingGemma2Model. It is used to instantiate a Embedding Gemma2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## EmbeddingGemma2VideoProcessor[[transformers.EmbeddingGemma2VideoProcessor]]

#### transformers.EmbeddingGemma2VideoProcessor[[transformers.EmbeddingGemma2VideoProcessor]]

```python
transformers.EmbeddingGemma2VideoProcessor(**kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/video_processing_embedding_gemma2.py#L173)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `None`) : Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `Resampling.BICUBIC`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.0, 0.0, 0.0]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[1.0, 1.0, 1.0]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

do_sample_frames (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to sample frames from the video before processing or to process the whole video.

video_metadata (`Annotated[~video_utils.VideoMetadata | dict | list[dict | ~video_utils.VideoMetadata] | list[list[dict | ~video_utils.VideoMetadata]] | None, None]`, *kwargs*) : Metadata of the video containing information about total duration, fps and total number of frames. It will be used to sample frames from video or compute timestamps. Don't pass any metadata unless you are trying to decode the video manually before processing

fps (`Annotated[int | float | None, None]`, *kwargs*, defaults to `1`) : Target frames to sample per second when `do_sample_frames=True`.

num_frames (`Annotated[int | None, None]`, *kwargs*) : Maximum number of frames to sample when `do_sample_frames=True`.

return_metadata (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether to return video metadata or not. Video metadats is an object containing info about video duration, fps, decoding backend, etc.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

patch_size (`int`, *kwargs*, *optional*) : Size of each image patch in pixels.

max_soft_tokens (`int`, *kwargs*, *optional*) : Maximum number of soft (vision) tokens per video frame. Must be one of {70, 140, 280, 560, 1120}.

pooling_kernel_size (`int`, *kwargs*, *optional*) : Spatial pooling kernel size applied after patchification.

add_timestamps (`bool`, *kwargs*, *optional*) : Whether to prefix each frame in the video placeholder expansion with its `mm:ss` timestamp. Requires `VideoMetadata` with a valid `fps`, since timestamps cannot be inferred from already-decoded frames.

max_frames (`int`, *kwargs*, *optional*) : The maximum number of frames to sample. If set, the sampled indices will be uniformly re-sampled to fit the budget.

overflow_strategy (`str`, *kwargs*, *optional*) : The strategy used to cut the total number of sampled frames down to fit into the budget. Can be set only to "uniform" or "truncate". Applied after FPS-based sampling, and on its own when FPS-based sampling is off or not applicable.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.19.0/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

Constructs a EmbeddingGemma2VideoProcessor video processor.

#### preprocess[[transformers.EmbeddingGemma2VideoProcessor.preprocess]]

```python
preprocess(videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]]], **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/video_processing_embedding_gemma2.py#L237)

## EmbeddingGemma2Processor[[transformers.EmbeddingGemma2Processor]]

#### transformers.EmbeddingGemma2Processor[[transformers.EmbeddingGemma2Processor]]

```python
transformers.EmbeddingGemma2Processor(feature_extractor, image_processor, tokenizer, video_processor, chat_template = None, image_seq_length: int = 280, audio_seq_length: int = 750, audio_ms_per_token: int = 40, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/processing_embedding_gemma2.py#L55)

**Parameters:**

feature_extractor (*Gemma4AudioFeatureExtractor*) : The feature extractor is a required input.

image_processor (*Gemma4ImageProcessor*) : The image processor is a required input.

tokenizer (*tokenizer_class*) : The tokenizer is a required input.

video_processor (*EmbeddingGemma2VideoProcessor*) : The video processor is a required input.

chat_template (*str*) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

image_seq_length (*int*, *optional*, defaults to 280) : The number of soft tokens per image used for placeholder expansion.

audio_seq_length (*int*, *optional*, defaults to 750) : The maximum number of audio soft tokens per audio segment. Serves as an upper-bound cap when dynamic audio token counts are computed.

audio_ms_per_token (*int*, *optional*, defaults to 40) : Milliseconds of audio per output soft token. Used to dynamically compute the number of audio placeholder tokens as `ceil(duration_ms / audio_ms_per_token)`. The default of 40 comes from the SSCP convolution's 4× time reduction on 10ms frames.

Constructs a EmbeddingGemma2Processor which wraps a feature extractor, a image processor, a tokenizer, and a video processor into a single processor.

[*EmbeddingGemma2Processor*] offers all the functionalities of [*Gemma4AudioFeatureExtractor*], [*Gemma4ImageProcessor*], [*tokenizer_class*], and [*EmbeddingGemma2VideoProcessor*]. See the
[*~Gemma4AudioFeatureExtractor*], [*~Gemma4ImageProcessor*], [*~tokenizer_class*], and [*~EmbeddingGemma2VideoProcessor*] for more information.

#### __call__[[transformers.EmbeddingGemma2Processor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/processing_utils.py#L657)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.19.0/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

## EmbeddingGemma2PreTrainedModel[[transformers.EmbeddingGemma2PreTrainedModel]]

#### transformers.EmbeddingGemma2PreTrainedModel[[transformers.EmbeddingGemma2PreTrainedModel]]

```python
transformers.EmbeddingGemma2PreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L442)

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.EmbeddingGemma2PreTrainedModel.forward]]

```python
forward(*args, **kwargs)
```

A mock value for a dotted path (e.g. `torch.float32`): attribute access chains,
calls behave as pass-through decorators, `repr` is the dotted path, and using it
as a base class substitutes a plain-`type` base (PEP 560 `__mro_entries__`), so
real subclasses keep a normal metaclass and `inspect.signature` reads their real
`__init__` instead of a mock's.

## EmbeddingGemma2TextModel[[transformers.EmbeddingGemma2TextModel]]

#### transformers.EmbeddingGemma2TextModel[[transformers.EmbeddingGemma2TextModel]]

```python
transformers.EmbeddingGemma2TextModel(config: EmbeddingGemma2TextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L477)

**Parameters:**

config ([EmbeddingGemma2TextConfig](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2TextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The EmbeddingGemma 2 text backbone. It owns the `embedding_projection` that maps the final hidden states
down to `config.embedding_dim`, so that the composite model needs no `forward` override.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.EmbeddingGemma2TextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L510)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

**Returns:** [BaseModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EmbeddingGemma2Config](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Config)) and inputs.

The [EmbeddingGemma2TextModel](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## EmbeddingGemma2Model[[transformers.EmbeddingGemma2Model]]

#### transformers.EmbeddingGemma2Model[[transformers.EmbeddingGemma2Model]]

```python
transformers.EmbeddingGemma2Model(config: EmbeddingGemma2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L633)

**Parameters:**

config ([EmbeddingGemma2Config](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The EmbeddingGemma 2 model: a vision backbone, an audio backbone and a text backbone whose final hidden
states are projected to `config.text_config.embedding_dim`. Intended to be wrapped by SentenceTransformers'
mean pooling and normalization.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.EmbeddingGemma2Model.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, input_features_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, image_position_ids: typing.Optional[torch.LongTensor] = None, video_position_ids: typing.Optional[torch.LongTensor] = None, num_frames_per_video: typing.Optional[torch.LongTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L741)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Gemma4ImageProcessor](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4ImageProcessor). See `Gemma4ImageProcessor.__call__()` for details ([EmbeddingGemma2Processor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Processor) uses [Gemma4ImageProcessor](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4ImageProcessor) for processing images).

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [EmbeddingGemma2VideoProcessor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2VideoProcessor). See `EmbeddingGemma2VideoProcessor.__call__()` for details ([EmbeddingGemma2Processor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Processor) uses [EmbeddingGemma2VideoProcessor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2VideoProcessor) for processing videos).

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [Gemma4AudioFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor). See [Gemma4AudioFeatureExtractor.__call__()](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor.__call__) for details ([EmbeddingGemma2Processor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Processor) uses [Gemma4AudioFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4AudioFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

input_features_mask (`torch.FloatTensor` of shape `(num_images, seq_length)`) : The attention mask for the input audio.

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*) : 2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding. Passed through to the vision encoder for positional embedding computation.

video_position_ids (`torch.LongTensor` of shape `(total_num_frames, max_patches, 2)`, *optional*) : 2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding. Passed through to the vision encoder for positional embedding computation.

num_frames_per_video (`torch.LongTensor` of shape `(num_videos,)`, *optional*) : Number of frames belonging to each video. Required whenever `pixel_values_videos` is passed, since the frames of all videos are concatenated along a single axis.

**Returns:** `EmbeddingGemma2ModelOutput` or `tuple(torch.FloatTensor)`

A `EmbeddingGemma2ModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EmbeddingGemma2Config](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Config)) and inputs.

The [EmbeddingGemma2Model](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
- **audio_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.

#### get_image_features[[transformers.EmbeddingGemma2Model.get_image_features]]

```python
get_image_features(pixel_values: FloatTensor, image_position_ids: typing.Optional[torch.LongTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L664)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [Gemma4ImageProcessor](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4ImageProcessor). See `Gemma4ImageProcessor.__call__()` for details ([EmbeddingGemma2Processor](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Processor) uses [Gemma4ImageProcessor](/docs/transformers/v5.19.0/en/model_doc/gemma4#transformers.Gemma4ImageProcessor) for processing images).

image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*) : The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EmbeddingGemma2Config](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Config)) and inputs.

Projects the last hidden state from the vision model into language model space.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

#### get_video_features[[transformers.EmbeddingGemma2Model.get_video_features]]

```python
get_video_features(pixel_values_videos: FloatTensor, video_position_ids: typing.Optional[torch.LongTensor] = None, num_frames_per_video: typing.Optional[torch.LongTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L882)

**Parameters:**

pixel_values_videos (`torch.FloatTensor` of shape `(total_num_frames, max_patches, patch_pixels)`) : The frames of every video in the batch, concatenated along the frame axis rather than stacked on a separate video axis, so that videos of different lengths can be batched together.

video_position_ids (`torch.LongTensor` of shape `(total_num_frames, max_patches, 2)`, *optional*) : 2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding. Passed through to the vision encoder for positional embedding computation.

num_frames_per_video (`torch.LongTensor` of shape `(num_videos,)`) : Number of frames belonging to each video, used to split the flat frame sequence back per video.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EmbeddingGemma2Config](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Config)) and inputs.

Projects the last hidden state from the vision encoder into language model space.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

#### get_audio_features[[transformers.EmbeddingGemma2Model.get_audio_features]]

```python
get_audio_features(input_features: Tensor, input_features_mask: Tensor, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py#L857)

**Parameters:**

input_features (`torch.FloatTensor` of shape `(num_images, seq_length, num_features)`) : The tensors corresponding to the input audio.

input_features_mask (`torch.FloatTensor` of shape `(num_images, seq_length)`) : The attention mask for the input audio.

**Returns:** `EmbeddingGemma2AudioModelOutput` or `tuple(torch.FloatTensor)`

A `EmbeddingGemma2AudioModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EmbeddingGemma2Config](/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2#transformers.EmbeddingGemma2Config)) and inputs.

Projects the last hidden state from the audio encoder into language model space.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **attention_mask** (`torch.BoolTensor`, *optional*) -- A torch.BoolTensor of shape `(batch_size, num_frames)`. True for valid positions, False for padding.

### EuroBERT
https://huggingface.co/docs/transformers/v5.19.0/model_doc/eurobert.md
