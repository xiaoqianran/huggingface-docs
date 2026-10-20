# Quantization concepts

Quantization reduces the memory footprint and computational cost of large machine learning models like those found in the Transformers library. It achieves this by representing the model's weights and or activations with lower-precision data types (like 8-bit integers or int8) instead of the standard 32-bit floating-point (float32).

Reducing a model's precision offers several significant benefits:

- Smaller model size: Lower-precision data types require less storage space. An int8 model, for example, is roughly 4 times smaller than its float32 counterpart.
- Faster inference: Operations on lower-precision data types, especially integers, can be significantly faster on compatible hardware (CPUs and GPUs often have specialized instructions for int8 operations). This leads to lower latency.
- Reduced energy consumption: Faster computations and smaller memory transfers often translate to lower power usage.

The primary trade-off in quantization is *efficiency* vs. *accuracy*. Reducing precision saves resources but inevitably introduces small errors (quantization noise). The goal is to minimize this error using appropriate schemes (affine/symmetric), granularity (per-tensor/channel), and techniques (PTQ/QAT) so that the model's performance on its target task degrades as little as possible.

The sections below cover quantization schemes, granularity, and techniques.

## Quantization schemes

The core idea is to map the range of values found in the original float32 weights and activations to the much smaller range represented by int8 (typically $[-128, 127]$).

This section covers how some quantization techniques work.

    

### Affine quantization

The most common method is *affine quantization*. For a given float32 tensor (like a layer's weights), it finds the minimum $val_{min}$ and maximum $val_{max}$ values. This range $[val_{min}, val_{max}]$ is mapped to the int8 range $[q_{min}, q_{max}]$, which is typically $[-128, 127]$.

There are two main ways to perform this mapping, *symmetric* and *asymmetric*. The choice between symmetric and asymmetric quantization determines how the float32 range is mapped to the int8 range.

- Symmetric: This method assumes the original float32 range is symmetric around zero ( $[ -a, a ]$ ). This range is mapped symmetrically to the int8 range, for example, $[-127, 127]$. A key characteristic is that the float32 value $0.0$ maps directly to the int8 value $0$. This only requires one parameter, the **scale ( $S$ )**, to define the mapping. It can simplify computations, but it might be less accurate if the original data distribution isn't naturally centered around zero.
- Asymmetric (Affine): This method does not assume the data is centered around zero. It maps the exact range $[val_{min}, val_{max}]$ from float32 to the full int8 range, like $[-128, 127]$. This requires two parameters, a **scale ( $S$ )** and a **zero-point ( $Z$ )**.

    scale ( $S$ ): A positive float32 number representing the ratio between the float32 and the int8 range.

$$
S = \frac{val_{max} - val_{min}}{q_{max} - q_{min}}
$$

zero-point ( $Z$ ): An int8 value that corresponds to the float32 value $0.0$.

$$
Z = q_{min} - round\left(\frac{val_{min}}{S}\right)
$$

> [!TIP]
> In symmetric quantization, Z would typically be fixed at 0.

With these parameters, a float32 value, $x$ can be quantized to int8 ( $q$ ) with the formula below.

$$
q = round\left(\frac{x}{S} + Z\right)
$$

The int8 value, $q$, can be dequantized back to approximate float32 with the formula below.

$$
x \approx S \cdot (q - Z)
$$

    

During inference, computations like matrix multiplication are performed using the int8 values ( $q$ ), and the result is dequantized back to float32 (often using a higher-precision accumulation type like int32 internally) before it is passed to the next layer.

### int4 and weight packing

    

int4 quantization further reduces the model size and memory usage (halving it compared to int8). The same affine or symmetric quantization principles apply, mapping the float32 range to the 16 possible values representable by int4 ( $[-8, 7]$ for signed int4).

A key aspect of int4 quantization is **weight packing**. Since most hardware can't natively handle 4-bit data types in memory, two int4 values are typically packed together into a single int8 byte for storage and transfer. For example, the first value might occupy the lower 4 bits and the second value the upper 4 bits of the byte (`packed_byte = (val1 & 0x0F) | (val2 << 4)`).

int4 is still beneficial even without native int4 compute because the primary benefit comes from reduced memory bandwidth. Loading packed int4 weights (stored as int8) from memory (RAM or VRAM) to the compute units is twice as fast as loading int8 weights. For large models, memory access is often a significant bottleneck. The speed up from faster data transfer can outweigh the computational overhead of unpacking and dequantizing on the fly, leading to overall faster inference, especially in memory-bound scenarios.

However, int4 quantization typically results in a larger accuracy drop compared to int8. Advanced quantization techniques like [GPTQ](./gptq) or [AWQ](./awq) are often necessary for good performance with int4.

### FP8 Quantization (A8W8)

    

A newer datatype, 8-bit floating-point (FP8), offers another way to reduce precision while retaining more accuracy than int8 in certain scenarios. FP8 keeps the floating-point structure (sign, exponent, mantissa) but uses fewer bits.

There are two common FP8 variants.

- E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits. Offers higher precision (more mantissa bits) but a smaller dynamic range (fewer exponent bits).
- E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits. Offers a wider dynamic range but lower precision.

FP8 is used in the *A8W8* quantization scheme, which quantizes both activations (A) and weights (W) to 8-bit precision.

While int8 has broad support, efficient FP8 computation requires specific hardware capabilities found in newer GPUs like NVIDIA H100/H200/B100, AMD Instinct MI300 series, and Intel XPU. Without native hardware acceleration, the benefits of FP8 might not be fully realized.

Transformers supports FP8 through specific backends like [FBGEMM](./fbgemm_fp8), [FineGrainedFP8](./finegrained_fp8), and [compressed-tensors](./compressed_tensors). These backends handle the underlying FP8 conversion and computation when the appropriate hardware and configurations are used.

## Granularity

Quantization parameters ( $S$ and $Z$) can be calculated in one of two ways.

- Per-Tensor: One set of $S$ and $Z$ for the entire tensor. Simpler, but less accurate if data values vary greatly within the tensor.
- Per-Channel (or Per-Group/Block): Separate $S$ and $Z$ for each channel or group. More accurate and better performance at the cost of slightly more complexity and memory.

    

## Quantization techniques

There are two main types of quantization techniques.

- Post-Training Quantization (PTQ): Quantization is applied  *after* the model is fully trained.
- Quantization-Aware Training (QAT): Quantization effects are simulated *during* training by inserting "fake quantization" ops that simulate the rounding errors of quantization. This lets the model adapt to quantization, and usually results in better accuracy, especially at lower bit-widths.

## Quantization in Transformers

Transformers integrates several quantization backends such as bitsandbytes, torchao, compressed-tensors, and more (refer to the quantization [overview](./overview) for more backends).

All backends are unified under the `HfQuantizer` API and associated `QuantizationConfig` classes. You can integrate your own custom quantization backends by implementing a custom `HfQuantizer` and `QuantizationConfig`, as shown in the [Contribution](./contribute) guide.

The typical workflow for quantization in Transformers is to:

1. Choose a quantization method suitable for your hardware and use case (see the [Overview](./overview) or [Selecting a quantization method](./selecting) guide to help you).
2. Load a pre-quantized model from the Hugging Face Hub or load a float32/float16/bfloat16 model and apply a specific quantization method with `QuantizationConfig`.

The example below demonstrates loading a 8B parameter model and quantizing it to 4-bits with bitsandbytes.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

## Resources

To explore quantization and related performance optimization concepts more deeply, check out the following resources.

- [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
- [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth)
- [Introduction to Quantization cooked in 🤗 with 💗🧑‍🍳](https://huggingface.co/blog/merve/quantization)
- [EfficientML.ai Lecture 5 - Quantization Part I](https://www.youtube.com/watch?v=RP23-dRVDWM)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Accelerating Generative AI with PyTorch Part 2: LLM Optimizations](https://pytorch.org/blog/accelerating-generative-ai-2/)

### Kernels
https://huggingface.co/docs/transformers/v5.20.2/main_classes/kernels.md

## Kernels

This page documents the kernels configuration utilities.

### kernelize[[transformers.kernelize]]

#### transformers.kernelize[[transformers.kernelize]]

```python
transformers.kernelize(model: PreTrainedModel, mode: Mode | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/integrations/hub_kernels.py#L953)

Temporarily register hidden kernel wrappers so `kernelize` can discover and replace them.

### KernelConfig[[transformers.KernelConfig]]

#### transformers.KernelConfig[[transformers.KernelConfig]]

```python
transformers.KernelConfig(kernel_mapping = None, use_local_kernel = False, inherit_mapping = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/utils/kernel_config.py#L89)

Kernel configuration class. This class is used to configure the kernel mapping for a model.

#### create_compatible_mapping[[transformers.KernelConfig.create_compatible_mapping]]

```python
create_compatible_mapping(model, compile = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/utils/kernel_config.py#L245)

Transforms a simple kernel_mapping of the form:
{
"RMSNorm":
("kernels-community/layer_norm:LlamaRMSNorm", {"version": 1, "trust_remote_code": True}),
...
},

or for local path:

{
"RMSNorm":
"/home/user/liger_kernels:LigerRMSNorm",
...
},

into a nested mapping:

{
"RMSNorm": {
"cuda": {
Mode.INFERENCE: LayerRepository(
repo_id="kernels-community/layer_norm",
layer_name="LlamaRMSNorm",
version=1,
trust_remote_code=True,
)
}
}
}

or for local path:

{
"RMSNorm": {
"cuda": {
Mode.INFERENCE: LocalLayerRepository(
repo_path=Path("/home/user/liger_kernels"),
layer_name="LigerRMSNorm",
)
}
}
}

that's compatible with the kernels library.

The device is inferred from the model's parameters if not provided.
The Mode is inferred from the model's training state.

#### sanitize_kernel_mapping[[transformers.KernelConfig.sanitize_kernel_mapping]]

```python
sanitize_kernel_mapping(model)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/utils/kernel_config.py#L128)

**Parameters:**

model : The model instance whose modules are checked for registered kernel_layer_name attributes.

**Raises:** ``ValueError``

- ``ValueError`` -- If a layer_name is not registered in the model, if a device is not supported,
  or if a repo_name is not a valid 'org/repo:layer_name' string.

Validates the kernel_mapping to ensure that:
1. Each layer_name in the mapping is registered in the model (i.e., the model contains a module with a matching kernel_layer_name).
2. Each kernel value is
   - either a string of the form 'org/repo:layer_name' or a tuple with the same as string and a dict of {"revision"/"version/trust_remote_code": ...},
   - or a dict mapping device types ("cuda", "rocm", "xpu", "npu") to such values as above.
3. Each device key in a dict is one of "cuda", "rocm", "xpu", or "npu".
5. Each trust remote code key must be a bool.
6. Each revision or version key must exist mutually exclusive if it has been passed explicitly.
7. Each repo_name is a valid repository and layer name in the format 'org/repo:layer_name' (i.e., a string containing both a slash and a colon).
8. If a local path is detected, it should be in the format '/abs/path:layer_name', where the absolute path points to the kernel repository, like "/home/user/layer_norm".

### Tokenizer
https://huggingface.co/docs/transformers/v5.20.2/main_classes/tokenizer.md
