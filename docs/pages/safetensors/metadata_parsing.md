# Metadata Parsing

Given the simplicity of the format, it's very simple and efficient to fetch and parse metadata about Safetensors weights – i.e. the list of tensors, their types, and their shapes or numbers of parameters – using small [(Range) HTTP requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests).

This parsing has been implemented in JS in [`huggingface.js`](https://huggingface.co/docs/huggingface.js/main/en/hub/modules#parsesafetensorsmetadata) (sample code follows below), but it would be similar in any language.

## Example use case

There can be many potential use cases. For instance, we use it on the HuggingFace Hub to display info about models which have safetensors weights:

    
    

    
    

## Usage

From [🤗 Hub](hf.co/models), you can get metadata of a model with [HTTP range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests) instead of downloading the entire safetensors file with all the weights. In this example python script below (you can use any language that has HTTP requests support), we are parsing metadata of [gpt2](https://huggingface.co/gpt2/blob/main/model.safetensors). 

```python
import requests # pip install requests
import struct

def parse_single_file(url):
    # Fetch the first 8 bytes of the file
    headers = {'Range': 'bytes=0-7'}
    response = requests.get(url, headers=headers)
    # Interpret the bytes as a little-endian unsigned 64-bit integer
    length_of_header = struct.unpack('<Q', response.content)[0]
    # Fetch length_of_header bytes starting from the 9th byte
    headers = {'Range': f'bytes=8-{7 + length_of_header}'}
    response = requests.get(url, headers=headers)
    # Interpret the response as a JSON object
    header = response.json()
    return header

url = "https://huggingface.co/gpt2/resolve/main/model.safetensors"
header = parse_single_file(url)

print(header)
# {
#   "__metadata__": { "format": "pt" },
#   "h.10.ln_1.weight": {
#     "dtype": "F32",
#     "shape": [768],
#     "data_offsets": [223154176, 223157248]
#   },
#   ...
# }
```

Using [`huggingface.js`](https://huggingface.co/docs/huggingface.js)

```ts
import { parseSafetensorsMetadata } from "@huggingface/hub";

const info = await parseSafetensorsMetadata({
	repo: { type: "model", name: "bigscience/bloom" },
});

console.log(info)
// {
//   sharded: true,
//   index: {
//     metadata: { total_size: 352494542848 },
//     weight_map: {
//       'h.0.input_layernorm.bias': 'model_00002-of-00072.safetensors',
//       ...
//     }
//   },
//   headers: {
//     __metadata__: {'format': 'pt'},
//     'h.2.attn.c_attn.weight': {'dtype': 'F32', 'shape': [768, 2304], 'data_offsets': [541012992, 548090880]},
//     ...
//   }
// }
```

Depending on whether the safetensors weights are sharded into multiple files or not, the output of the call above will be:

```ts
export type SafetensorsParseFromRepo =
| {
		sharded: false;
		header: SafetensorsFileHeader;
	}
| {
		sharded: true;
		index: SafetensorsIndexJson;
		headers: SafetensorsShardedHeaders;
	};
```

where the underlying `types` are the following:

```ts
type FileName = string;

type TensorName = string;
type Dtype = "F64" | "F32" | "F16" | "BF16" | "I64" | "I32" | "I16" | "I8" | "U8" | "BOOL";

interface TensorInfo {
	dtype: Dtype;
	shape: number[];
	data_offsets: [number, number];
}

type SafetensorsFileHeader = Record<TensorName, TensorInfo> & {
	__metadata__: Record<string, string>;
};

interface SafetensorsIndexJson {
	weight_map: Record<TensorName, FileName>;
}

export type SafetensorsShardedHeaders = Record<FileName, SafetensorsFileHeader>;
```

[`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) provides a Python API to parse safetensors metadata.
Use [`get_safetensors_metadata`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.get_safetensors_metadata) to get all safetensors metadata of a model.
Depending on if the model is sharded or not, one or multiple safetensors files will be parsed.

```python
>>> from huggingface_hub import get_safetensors_metadata

# Parse repo with single weights file
>>> metadata = get_safetensors_metadata("bigscience/bloomz-560m")
>>> metadata
SafetensorsRepoMetadata(
    metadata=None,
    sharded=False,
    weight_map={'h.0.input_layernorm.bias': 'model.safetensors', ...},
    files_metadata={'model.safetensors': SafetensorsFileMetadata(...)}
)
>>> metadata.files_metadata["model.safetensors"].metadata
{'format': 'pt'}

# Parse repo with sharded model (i.e. multiple weights files)
>>> metadata = get_safetensors_metadata("bigscience/bloom")
Parse safetensors files: 100%|██████████████████████████████████████████| 72/72 [00:12<00:00,  5.78it/s]
>>> metadata
SafetensorsRepoMetadata(metadata={'total_size': 352494542848}, sharded=True, weight_map={...}, files_metadata={...})
>>> len(metadata.files_metadata)
72  # All safetensors files have been fetched

# Parse repo that is not a safetensors repo
>>> get_safetensors_metadata("runwayml/stable-diffusion-v1-5")
NotASafetensorsRepoError: 'runwayml/stable-diffusion-v1-5' is not a safetensors repo. Couldn't find 'model.safetensors.index.json' or 'model.safetensors' files.
```

To parse the metadata of a single safetensors file, use [`parse_safetensors_file_metadata`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.parse_safetensors_file_metadata).

## Example output

For instance, here are the number of params per dtype for a few models on the HuggingFace Hub. Also see [this issue](https://github.com/huggingface/safetensors/issues/44) for more examples of usage.

model | safetensors | params
--- | --- | ---
[gpt2](https://huggingface.co/gpt2?show_tensors=true) | single-file | { 'F32' => 137022720 }
[roberta-base](https://huggingface.co/roberta-base?show_tensors=true) | single-file | { 'F32' => 124697433, 'I64' => 514 }
[Jean-Baptiste/camembert-ner](https://huggingface.co/Jean-Baptiste/camembert-ner?show_tensors=true) | single-file | { 'F32' => 110035205, 'I64' => 514 }
[roberta-large](https://huggingface.co/roberta-large?show_tensors=true) | single-file | { 'F32' => 355412057, 'I64' => 514 }
[distilbert-base-german-cased](https://huggingface.co/distilbert-base-german-cased?show_tensors=true) | single-file | { 'F32' => 67431550 }
[EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b?show_tensors=true) | sharded | { 'F16' => 20554568208, 'U8' => 184549376 }
[bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m?show_tensors=true) | single-file | { 'F16' => 559214592 }
[bigscience/bloom](https://huggingface.co/bigscience/bloom?show_tensors=true) | sharded | { 'BF16' => 176247271424 }
[bigscience/bloom-3b](https://huggingface.co/bigscience/bloom-3b?show_tensors=true) | single-file | { 'F16' => 3002557440 }

### Safetensors
https://huggingface.co/docs/safetensors/main/index.md

# Safetensors

Safetensors is a new simple format for storing tensors safely (as opposed to pickle) and that is still fast (zero-copy). Safetensors is really [fast 🚀](./speed).

## Installation

with pip:
```
pip install safetensors
```

with conda:
```
conda install -c conda-forge safetensors
```

## Usage

### Load tensors

```python
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
```

Loading only part of the tensors (interesting when running on multiple GPU)

```python
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    tensor_slice = f.get_slice("embedding")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]
```

### Save tensors

```python
import torch
from safetensors.torch import save_file

tensors = {
    "embedding": torch.zeros((2, 2)),
    "attention": torch.zeros((2, 3))
}
save_file(tensors, "model.safetensors")
```

## Format

Let's say you have safetensors file named `model.safetensors`, then `model.safetensors` will have the following internal format:

    

## Featured Projects

Safetensors is being used widely at leading AI enterprises, such as [Hugging Face](https://huggingface.co/), [EleutherAI](https://www.eleuther.ai/), and [StabilityAI](https://stability.ai/). Here is a non-exhaustive list of projects that are using safetensors:

* [huggingface/transformers](https://github.com/huggingface/transformers)
* [ml-explore/mlx](https://github.com/ml-explore/mlx)
* [huggingface/candle](https://github.com/huggingface/candle)
* [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [Llama-cpp](https://github.com/ggerganov/llama.cpp/blob/e6a46b0ed1884c77267dc70693183e3b7164e0e0/convert.py#L537)
* [microsoft/TaskMatrix](https://github.com/microsoft/TaskMatrix)
* [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI)
* [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
* [CivitAI](https://civitai.com/)
* [huggingface/diffusers](https://github.com/huggingface/diffusers)
* [coreylowman/dfdx](https://github.com/coreylowman/dfdx)
* [invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI)
* [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)
* [Sanster/lama-cleaner](https://github.com/Sanster/lama-cleaner)
* [PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
* [AIGC-Audio/AudioGPT](https://github.com/AIGC-Audio/AudioGPT)
* [brycedrennan/imaginAIry](https://github.com/brycedrennan/imaginAIry)
* [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
* [alvarobartt/safejax](https://github.com/alvarobartt/safejax)
* [MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)
* [rachthree/safestructures](https://github.com/rachthree/safestructures)
* [justinchuby/onnx-safetensors](https://github.com/justinchuby/onnx-safetensors)

### Speed Comparison
https://huggingface.co/docs/safetensors/main/speed.md
