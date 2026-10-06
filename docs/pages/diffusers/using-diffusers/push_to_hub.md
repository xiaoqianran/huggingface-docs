# Sharing pipelines and models

Share your pipelines, models, and schedulers on the Hub with [PushToHubMixin](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin). This mixin:

1. creates a repository on the Hub
2. saves your model, scheduler, or pipeline files so they can be reloaded later
3. uploads the folder containing these files to the Hub

Log in to your Hugging Face account with your access [token](https://huggingface.co/settings/tokens).

```py
from huggingface_hub import notebook_login

notebook_login()
```

```bash
hf auth login
```

Push to your user namespace with a short id (`"my-controlnet-model"`) or to an org with `"your-org/my-controlnet-model"`.

## Models

To push a model to the Hub, call [push_to_hub()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) and specify the repository id of the model.

```py
from diffusers import ControlNetModel

controlnet = ControlNetModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    in_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    cross_attention_dim=32,
    conditioning_embedding_out_channels=(16, 32),
)
controlnet.push_to_hub("my-controlnet-model")
```

The [push_to_hub()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) method saves the model's `config.json` file and the weights are automatically saved as [safetensors files](./other-formats#safetensors).

Load the model again with [ControlNetModel.from_pretrained()](/docs/diffusers/v0.41.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained).

```py
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model")
```

## Scheduler

To push a scheduler to the Hub, call [push_to_hub()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) and specify the repository id of the scheduler.

```py
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
scheduler.push_to_hub("my-ddim-scheduler")
```

The [push_to_hub()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) method saves the scheduler's `scheduler_config.json` file to the specified repository.

Load the scheduler again with [from_pretrained()](/docs/diffusers/v0.41.0/en/api/schedulers/overview#diffusers.SchedulerMixin.from_pretrained).

```py
scheduler = DDIMScheduler.from_pretrained("your-namespace/my-ddim-scheduler")
```

## Pipeline

To push a pipeline to the Hub, load it with [from_pretrained()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained), then call [push_to_hub()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) with a repository id.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipeline.push_to_hub("your-namespace/my-qwen-image")
```

The [push_to_hub()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) method saves each component to a subfolder in the repository. Load the pipeline again with [DiffusionPipeline.from_pretrained()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained).

```py
pipeline = DiffusionPipeline.from_pretrained("your-namespace/my-qwen-image")
```

## Privacy

Set `private=True` in [push_to_hub()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.utils.PushToHubMixin.push_to_hub) to keep a model, scheduler, or pipeline files private.

```py
controlnet.push_to_hub("my-controlnet-model-private", private=True)
```

Pass `create_pr=True` to open a pull request on an existing Hub repository instead of pushing straight to the default branch.

Models and pipelines also accept `variant=` on push when you want a named weight file such as `fp16`. Schedulers do not use `variant`.

Private repositories are only visible to you. Other users won't be able to clone the repository and it won't appear in search results. Even if a user has the URL to your private repository, they'll receive a `404 - Sorry, we can't find the page you are looking for`. You must be [logged in](https://huggingface.co/docs/huggingface_hub/quick-start#login) to load a model from a private repository.

### Command line interface
https://huggingface.co/docs/diffusers/v0.41.0/using-diffusers/cli.md
