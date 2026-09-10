# Torch shared tensors

## TL;DR

Using specific functions, which should work in most cases for you.
This is not without side effects.

```python
from safetensors.torch import load_model, save_model

save_model(model, "model.safetensors")
# Instead of save_file(model.state_dict(), "model.safetensors")

load_model(model, "model.safetensors")
# Instead of model.load_state_dict(load_file("model.safetensors"))
```

## What are shared tensors ?

Pytorch uses shared tensors for some computation.
This is extremely interesting to reduce memory usage in general.

One very classic use case is in transformers the `embeddings` are shared with
`lm_head`. By using the same matrix, the model uses less parameters, and gradients 
flow much better to the `embeddings` (which is the start of the model, so they don't
flow easily there, whereas `lm_head` is at the tail of the model, so gradients are
extremely good over there, since they are the same tensors, they both benefit)

```python
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(100, 100)
        self.b = self.a

    def forward(self, x):
        return self.b(self.a(x))

model = Model()
print(model.state_dict())
# odict_keys(['a.weight', 'a.bias', 'b.weight', 'b.bias'])
torch.save(model.state_dict(), "model.bin")
# This file is now 41k instead of ~80k, because A and B are the same weight hence only 1 is saved on disk with both `a` and `b` pointing to the same buffer
```

## Why are shared tensors not saved in `safetensors` ?

Multiple reasons for that:

- *Not all frameworks support them* for instance `tensorflow` does not.
  So if someone saves shared tensors in torch, there is no way to 
  load them in a similar fashion so we could not keep the same `Dict[str, Tensor]`
  API.
- *It makes lazy loading very quickly.*
  Lazy loading is the ability to load only some tensors, or part of tensors for
  a given file. This is trivial to do without sharing tensors but with tensor sharing

  ```python
  with safe_open("model.safetensors", framework="pt") as f:
      a = f.get_tensor("a")
      b = f.get_tensor("b")
  ```

  Now it's impossible with this given code to "reshare" buffers after the fact.
  Once we give the `a` tensor we have no way to give back the same memory when
  you ask for `b`. (In this particular example we could keep track of given buffers
  but this is not the case in general, since you could do arbitrary work with `a`
  like sending it to another device before asking for `b`)
- *It can lead to much larger file than necessary*.
  If you are saving a shared tensor which is only a fraction of a larger tensor,
  then saving it with pytorch leads to saving the entire buffer instead of saving
  just what is needed.

  ```python
  a = torch.zeros((100, 100))
  b = a[:1, :]
  torch.save({"b": b}, "model.bin")
  # File is 41k instead of the expected 400 bytes
  # In practice it could happen that you save several 10GB instead of 1GB.
  ```

Now with all those reasons being mentioned, nothing is set in stone in there.
Shared tensors do not cause unsafety, or denial of service potential, so this
decision could be revisited if current workarounds are not satisfactory.

## How does it work ?

The design is rather simple.
We're going to look for all shared tensors, then looking for all tensors
covering the entire buffer (there can be multiple such tensors).
That gives us multiple names which can be saved, we simply choose the first one

During `load_model`, we are loading a bit like `load_state_dict` does, except
we're looking into the model itself, to check for shared buffers, and ignoring
the "missed keys" which were actually covered by virtue of buffer sharing (they
were properly loaded since there was a buffer that loaded under the hood).
Every other error is raised as-is

**Caveat**: This means we're dropping some keys within the file. meaning if you're
checking for the keys saved on disk, you will see some "missing tensors" or if you're
using `load_state_dict`. Unless we start supporting shared tensors directly in
the format there's no real way around it.

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

### Flax API[[safetensors.flax.load_file]]
https://huggingface.co/docs/safetensors/main/api/flax.md
