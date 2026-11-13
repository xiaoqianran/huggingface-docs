# Exporters

New export backends can be added to Transformers by subclassing `HfExporter` and registering it with
[register_backend()](/docs/transformers/v5.23.0/en/main_classes/exporters#transformers.exporters.register_backend).

Learn how to use the built-in exporters in the [Exporters](../exporters) guide.

## AutoHfExporter[[transformers.exporters.AutoHfExporter]]

#### transformers.exporters.AutoHfExporter[[transformers.exporters.AutoHfExporter]]

```python
transformers.exporters.AutoHfExporter()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/auto.py#L112)

The Auto-HF exporter class that takes care of automatically instantiating to the correct
`HfExporter` given the `ExportConfig`.

## AutoExportConfig[[transformers.exporters.AutoExportConfig]]

#### transformers.exporters.AutoExportConfig[[transformers.exporters.AutoExportConfig]]

```python
transformers.exporters.AutoExportConfig()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/auto.py#L101)

The Auto-HF export config class that takes care of automatically dispatching to the correct
export config given an export config stored in a dictionary.

## register_backend[[transformers.exporters.register_backend]]

#### transformers.exporters.register_backend[[transformers.exporters.register_backend]]

```python
transformers.exporters.register_backend(exporter: type[HfExporter], runner: type[ModelRunner] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/auto.py#L61)

Register an exporter, with its `config_class`, under its `export_format`, and the runner that runs its exports,
so the auto classes find them.

## HfExporter[[transformers.exporters.HfExporter]]

#### transformers.exporters.HfExporter[[transformers.exporters.HfExporter]]

```python
transformers.exporters.HfExporter()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L321)

Abstract base class for all Transformers exporters.

To add a backend, subclass and implement `~HfExporter.export_program` to trace one graph, and
`~HfExporter.save_program` / `~HfExporter.load_program` to write one out and read it back. The public
`~HfExporter.export` and `~HfExporter.export_for_generation` build an `ExportedModel` on top of
them.

#### export[[transformers.exporters.HfExporter.export]]

```python
export(model: PreTrainedModel, sample_inputs: MutableMapping[str, torch.Tensor | Cache], config: ExportConfigMixin | dict[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L407)

**Returns:** `ExportedModel`

A single `"model"` component; the backend's program is `.program`, and
`save_pretrained` writes it with its metadata and the model config.

Export the model as one graph.

Takes the same **forward** kwargs as `~HfExporter.export_program`.

#### export_for_generation[[transformers.exporters.HfExporter.export_for_generation]]

```python
export_for_generation(model: PreTrainedModel, sample_inputs: MutableMapping[str, torch.Tensor | Cache], config: ExportConfigMixin | dict[str, ExportConfigMixin], generation_config: GenerationConfig | None = None, multi_token_decode: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L426)

**Parameters:**

model ([PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel)) : The generative model to export. Must support `model.generate(**sample_inputs)`.

sample_inputs (`dict[str, torch.Tensor | Cache]`) : **Generate** kwargs — what you'd pass to `model.generate(**sample_inputs)` (typically `input_ids` + `attention_mask`, plus any modality inputs like `pixel_values` / `input_features` for multi-modal models). Per-stage forward kwargs are captured internally.

config (`ExportConfigMixin` or `dict[str, ExportConfigMixin]`) : Backend-specific configuration. Pass a single config to apply to every component, or a `dict` keyed by component name (e.g. `"image_encoder"`, `"language_model"`, `"lm_head"`, `"decode"`) to override per-component — all component names must be present in the dict.

generation_config ([GenerationConfig](/docs/transformers/v5.23.0/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) : Forwarded to the `generate()` capture (defaults to the model's own). Pass one with `cache_implementation="static"` to export against a fixed-size `StaticCache`.

multi_token_decode (`bool`, *optional*, defaults to `False`) : Whether the `decode` component processes multiple query tokens at once — a dynamic query axis (prefill on an empty cache, continuation-from-past otherwise) — vs the classic single-token step (see [decompose_for_generation()](/docs/transformers/v5.23.0/en/main_classes/exporters#transformers.exporters.utils.decompose_for_generation)). Only stays dynamic under a dynamic-shape export (`config.dynamic=True`).

**Returns:** `ExportedModel`

One `ExportedComponent` per stage, with the same keys as
[decompose_for_generation()](/docs/transformers/v5.23.0/en/main_classes/exporters#transformers.exporters.utils.decompose_for_generation).

Decompose a generative model and export each component independently.

Thin wrapper around [decompose_for_generation()](/docs/transformers/v5.23.0/en/main_classes/exporters#transformers.exporters.utils.decompose_for_generation) that calls
`~HfExporter.export_program` on every returned `(submodel, forward_inputs)` pair. If you need
the intermediate `(submodel, forward_inputs)` pairs (for verification, custom inputs,
skipping a stage, …), call [decompose_for_generation()](/docs/transformers/v5.23.0/en/main_classes/exporters#transformers.exporters.utils.decompose_for_generation) directly.

#### export_program[[transformers.exporters.HfExporter.export_program]]

```python
export_program(model: PreTrainedModel, sample_inputs: MutableMapping[str, torch.Tensor | Cache], config: ExportConfigMixin)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L351)

**Parameters:**

model ([PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel)) : The model to export.

sample_inputs (`dict[str, torch.Tensor | Cache]`) : **Forward** kwargs — what you'd pass to `model(**sample_inputs)`. These are used directly as the example inputs during tracing. For an autoregressive decode-step export, this means you need to include `past_key_values`, `cache_position`, etc. If you only have generation-style inputs, use `~HfExporter.export_for_generation` instead — it runs `model.generate` for you and exports each stage.

config (`ExportConfigMixin`) : Backend-specific configuration.

**Returns:** `tuple[Any, ExportMetadata]`

The backend's program object (`ExportedProgram`, `onnx_ir.Model`,
`ExecutorchProgramManager`, `openvino.Model`) and the metadata recorded while tracing it
(`build_export_metadata()`), returned separately since only some formats can
carry it.

Trace one graph. The backend extension point, paired with `~HfExporter.save_program`.

#### load_program[[transformers.exporters.HfExporter.load_program]]

```python
load_program(path: str | Path)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L385)

Read back a program written by `~HfExporter.save_program`. It can differ from what
`~HfExporter.export_program` returned: a runtime `Program` for ExecuTorch, whose `ExecutorchProgramManager`
can't be rebuilt from the file.

#### program_files[[transformers.exporters.HfExporter.program_files]]

```python
program_files(filename: str)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L392)

The files `~HfExporter.save_program` writes for a program saved as `filename`: the file itself, plus
any the format keeps beside it (weights, external data).

#### save_program[[transformers.exporters.HfExporter.save_program]]

```python
save_program(program: Any, path: str | Path)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L380)

Write one program returned by `~HfExporter.export_program` to `path`, in the backend's file format.

#### validate_environment[[transformers.exporters.HfExporter.validate_environment]]

```python
validate_environment()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L347)

Check `required_packages` are installed and warn on version drift from `tested_versions`.

## ExportedModel[[transformers.exporters.ExportedModel]]

#### transformers.exporters.ExportedModel[[transformers.exporters.ExportedModel]]

```python
transformers.exporters.ExportedModel(components: dict[str, ExportedComponent], export_format: ExportFormat | str, config: PreTrainedConfig | None = None, generation_config: GenerationConfig | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L67)

An exported model: its graphs (`components`, `{name: ExportedComponent}`) and the configs they were traced
against, called the way you'd call the model it came from.

`~HfExporter.export` produces a single `"model"` component, whose program is `.program`;
`~HfExporter.export_for_generation` produces one component per generation stage.

```python
>>> exported = OnnxExporter().export(model, inputs, OnnxConfig(dynamic=True))
>>> exported.save_pretrained("exported")
>>> outputs = ExportedModel.from_pretrained("exported", device="cuda")(**inputs)
>>> outputs.last_hidden_state
```

#### __call__[[transformers.exporters.ExportedModel.__call__]]

```python
__call__(share_inputs: bool = True, share_outputs: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L143)

Run a single-graph export, returning what the source model returns. Kwargs the export was not traced with
are dropped, so a processor's whole output can be passed. `share_inputs` and `share_outputs` go to the runner
(see `ModelRunner.__call__`): with `share_outputs`, the outputs are valid until the next call.

#### program[[transformers.exporters.ExportedModel.program]]

```python
program()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L108)

The backend's own program, for a single-graph export.

#### to[[transformers.exporters.ExportedModel.to]]

```python
to(device = None, **runner_kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L130)

Load the programs on their backend's runtime, on `device` (the backend's default when `None`).
`runner_kwargs` go to the backend's runner: `providers` and `session_options` for ONNX Runtime, `config`
(compile properties) and `torch_threads` (split the CPU with torch work around the model) for OpenVINO.

#### save_pretrained[[transformers.exporters.ExportedModel.save_pretrained]]

```python
save_pretrained(save_directory: str | Path, push_to_hub: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L156)

**Parameters:**

save_directory (`str` or `Path`) : The directory to write to, created if it does not exist.

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether to upload the saved files to the Hub too, to `repo_id` (defaults to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Passed to the Hub upload: `repo_id`, `commit_message`, `token`, `private`, ...

Write the components (one file each), the configs, and a manifest naming each component's file and the
metadata recorded when tracing it. `~ExportedModel.from_pretrained` loads them back.

#### from_pretrained[[transformers.exporters.ExportedModel.from_pretrained]]

```python
from_pretrained(pretrained_model_name_or_path: str | Path, device = None, cache_dir: str | os.PathLike | None = None, force_download: bool = False, local_files_only: bool = False, token: str | bool | None = None, revision: str = 'main', subfolder: str = '', trust_remote_code: bool = False, **runner_kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L206)

**Parameters:**

pretrained_model_name_or_path (`str` or `Path`) : A local directory, or a Hub repo id. From the Hub, only the files the manifest names are downloaded, so an export can sit next to the model's own weights.

device (`str` or `torch.device`, *optional*) : Where to run it (the backend's default when `None`).

cache_dir (`Union[str, os.PathLike]`, *optional*) : Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.

local_files_only (`bool`, *optional*, defaults to `False`) : Whether or not to only look at local files (i.e., do not try to download the model).

token (`str` or `bool`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.    To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.  

subfolder (`str`, *optional*, defaults to `""`) : In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here.

trust_remote_code (`bool`, *optional*, defaults to `False`) : Whether to load the config of a model whose code lives in its repo, as for [from_pretrained()](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoConfig.from_pretrained).

runner_kwargs (*optional*) : The backend's runner options, as for `~ExportedModel.to`: `providers` and `session_options` for ONNX Runtime, `config` and `torch_threads` for OpenVINO.

Load what `~ExportedModel.save_pretrained` wrote: each component through its backend's
`~HfExporter.load_program`, with its metadata, and the configs saved next to them. The programs are loaded
on their backend's runtime right away, ready to run.

## ExportedComponent[[transformers.exporters.ExportedComponent]]

#### transformers.exporters.ExportedComponent[[transformers.exporters.ExportedComponent]]

```python
transformers.exporters.ExportedComponent(program: Any, metadata: ExportMetadata)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L60)

One exported graph: the backend's program, and what the exporter recorded about it.

## ExportMetadata[[transformers.exporters.ExportMetadata]]

#### transformers.exporters.ExportMetadata[[transformers.exporters.ExportMetadata]]

```python
transformers.exporters.ExportMetadata(architecture: str | None = None, packages: dict[str, str] = <factory>, export_config: dict[str, Any] = <factory>, export_dtype: str | None = None, export_device: str | None = None, kwargs: dict[str, dict] = <factory>, in_spec: str | None = None, out_spec: str | None = None, in_constants: dict[str, Any] = <factory>, out_constants: list = <factory>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/metadata.py#L58)

What the exporter recorded about one graph (`build_export_metadata()`).

#### from_dict[[transformers.exporters.ExportMetadata.from_dict]]

```python
from_dict(values: Mapping[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/metadata.py#L75)

Build the metadata from its saved dict, skipping fields this class doesn't have (added by a newer version).

## ModelRunner[[transformers.exporters.ModelRunner]]

#### transformers.exporters.ModelRunner[[transformers.exporters.ModelRunner]]

```python
transformers.exporters.ModelRunner(metadata: ExportMetadata)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/runners/base.py#L29)

Runs one exported program on its backend: `runner(**kwargs)` returns what the source model returns.

#### from_program[[transformers.exporters.ModelRunner.from_program]]

```python
from_program(program: Any, metadata: ExportMetadata, device = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/runners/base.py#L56)

A runner for a program as an export returns it or `~HfExporter.load_program` reads it back, on
`device` (the backend's default when `None`).

#### unflatten[[transformers.exporters.ModelRunner.unflatten]]

```python
unflatten(outputs: list)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/runners/base.py#L47)

Rebuild the model's outputs from the program's flat ones (which come last) and the recorded constants.

## DynamoExporter[[transformers.exporters.DynamoExporter]]

#### transformers.exporters.DynamoExporter[[transformers.exporters.DynamoExporter]]

```python
transformers.exporters.DynamoExporter()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/exporter_dynamo.py#L69)

Exporter that converts a [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel) to an `ExportedProgram`.

Example:

```python
>>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

>>> exporter = DynamoExporter()
>>> exported = exporter.export(model, inputs, config=DynamoConfig(dynamic=True))
>>> outputs = exported(**inputs)
>>> exported.save_pretrained("exported")  # `model.pt2`, its metadata and the model config
```

#### export[[transformers.exporters.DynamoExporter.export]]

```python
export(model: PreTrainedModel, sample_inputs: MutableMapping[str, torch.Tensor | Cache], config: ExportConfigMixin | dict[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L407)

**Returns:** `ExportedModel`

A single `"model"` component; the backend's program is `.program`, and
`save_pretrained` writes it with its metadata and the model config.

Export the model as one graph.

Takes the same **forward** kwargs as `~HfExporter.export_program`.

## OnnxExporter[[transformers.exporters.OnnxExporter]]

#### transformers.exporters.OnnxExporter[[transformers.exporters.OnnxExporter]]

```python
transformers.exporters.OnnxExporter()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/exporter_onnx.py#L113)

Exporter that converts a [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel) to an ONNX graph (an `onnx_ir.Model`).

Example:

```python
>>> from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

>>> exporter = OnnxExporter()
>>> exported = exporter.export(model, inputs, config=OnnxConfig(dynamic=True))
>>> outputs = exported(**inputs)  # on ONNX Runtime
>>> onnx_model = exported.program  # an `onnx_ir.Model`
>>> exported.save_pretrained("exported")  # `model.onnx`, its metadata and the model config
```

#### export[[transformers.exporters.OnnxExporter.export]]

```python
export(model: PreTrainedModel, sample_inputs: MutableMapping[str, torch.Tensor | Cache], config: ExportConfigMixin | dict[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L407)

**Returns:** `ExportedModel`

A single `"model"` component; the backend's program is `.program`, and
`save_pretrained` writes it with its metadata and the model config.

Export the model as one graph.

Takes the same **forward** kwargs as `~HfExporter.export_program`.

## ExecutorchExporter[[transformers.exporters.ExecutorchExporter]]

#### transformers.exporters.ExecutorchExporter[[transformers.exporters.ExecutorchExporter]]

```python
transformers.exporters.ExecutorchExporter()
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/exporter_executorch.py#L107)

Exporter that converts a [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel) to an ExecuTorch `ExecutorchProgramManager`.

Example:

```python
>>> from transformers.exporters.exporter_executorch import ExecutorchExporter, ExecutorchConfig

>>> exporter = ExecutorchExporter()
>>> exported = exporter.export(model, inputs, config=ExecutorchConfig(backend="xnnpack"))
>>> outputs = exported(**inputs)  # on the ExecuTorch runtime
>>> exported.save_pretrained("exported")  # `model.pte`, its metadata and the model config
```

#### export[[transformers.exporters.ExecutorchExporter.export]]

```python
export(model: PreTrainedModel, sample_inputs: MutableMapping[str, torch.Tensor | Cache], config: ExportConfigMixin | dict[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/base.py#L407)

**Returns:** `ExportedModel`

A single `"model"` component; the backend's program is `.program`, and
`save_pretrained` writes it with its metadata and the model config.

Export the model as one graph.

Takes the same **forward** kwargs as `~HfExporter.export_program`.

## DynamoConfig[[transformers.exporters.DynamoConfig]]

#### transformers.exporters.DynamoConfig[[transformers.exporters.DynamoConfig]]

```python
transformers.exporters.DynamoConfig(export_format: ExportFormat = <ExportFormat.DYNAMO: 'dynamo'>, dynamic: bool = False, strict: bool = False, dynamic_shapes: dict[str, Any] | None = None, prefer_deferred_runtime_asserts_over_guards: bool = False, quantizer: ExportQuantizer | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/configs.py#L92)

**Parameters:**

dynamic (*bool*, *optional*, defaults to *False*) : Whether to export with dynamic (symbolic) shapes. When *True* and *dynamic_shapes* is not set, all tensor dimensions are set to *Dim.AUTO* automatically.

strict (*bool*, *optional*, defaults to *False*) : Whether to enable strict mode in *torch.export*. Runs the full symbolic trace and catches more errors, but is slower and more likely to fail on complex models.

dynamic_shapes (*dict[str, Any]*, *optional*) : Explicit per-input dynamic shape specifications passed to *torch.export*. Takes precedence over *dynamic*.

prefer_deferred_runtime_asserts_over_guards (*bool*, *optional*, defaults to *False*) : When *True*, data-dependent shape guards are emitted as runtime asserts in the exported graph instead of failing the export at trace time when a guard wouldn't hold across the full symbolic shape range. Most transformer LLMs need this set to *True* when using fine-grained `Dim(min=, max=)` bounds. Not needed with `dynamic=True` / `Dim.AUTO`, where `torch.export` infers shape relations instead of verifying them against the user-stated bounds.

quantizer ([*~exporters.quantizers.ExportQuantizer*], *optional*) : Post-training quantization applied during the export: [*~exporters.quantizers.PT2EQuantizer*] or [*~exporters.quantizers.NNCFTorchFXQuantizer*] on the FX graph for every backend, [*~exporters.quantizers.OnnxRuntimeQuantizer*] on the converted ONNX model, or [*~exporters.quantizers.NNCFOpenVINOQuantizer*] / [*~exporters.quantizers.NNCFOnnxQuantizer*] on the converted OpenVINO / ONNX model.

Configuration class for exporting models via *torch.export*.

## OnnxConfig[[transformers.exporters.OnnxConfig]]

#### transformers.exporters.OnnxConfig[[transformers.exporters.OnnxConfig]]

```python
transformers.exporters.OnnxConfig(export_format: ExportFormat = <ExportFormat.ONNX: 'onnx'>, dynamic: bool = False, strict: bool = False, dynamic_shapes: dict[str, Any] | None = None, prefer_deferred_runtime_asserts_over_guards: bool = False, quantizer: ExportQuantizer | None = None, opset_version: int | None = None, optimize: bool = True, ort_fusions: Literal['cpu', 'cuda', 'webgpu', 'dml'] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/configs.py#L134)

**Parameters:**

opset_version (`int`, *optional*) : ONNX opset version to target. Defaults to the latest opset supported by the installed `onnxscript` version.

optimize (`bool`, *optional*, defaults to `True`) : Run `onnxscript` optimisation passes (constant folding, dead-code elimination, …) on the exported graph. Disable for models that hit upstream `onnxscript` optimiser bugs.

ort_fusions (`str`, *optional*) : Fuse the graph into ONNX Runtime's ops for the execution provider it runs on: `"cpu"`, `"cuda"`, `"webgpu"` or `"dml"`. The fused graph only runs on ONNX Runtime. Requires `optimize=True`.

Configuration class for exporting models to ONNX via `torch.onnx.export`.

Inherits all fields from `DynamoConfig` (`dynamic`, `strict`,
`dynamic_shapes`, `prefer_deferred_runtime_asserts_over_guards`).

## ExecutorchConfig[[transformers.exporters.ExecutorchConfig]]

#### transformers.exporters.ExecutorchConfig[[transformers.exporters.ExecutorchConfig]]

```python
transformers.exporters.ExecutorchConfig(export_format: ExportFormat = <ExportFormat.EXECUTORCH: 'executorch'>, dynamic: bool = False, strict: bool = False, dynamic_shapes: dict[str, Any] | None = None, prefer_deferred_runtime_asserts_over_guards: bool = False, quantizer: ExportQuantizer | None = None, backend: str = 'xnnpack', alloc_graph_input: bool = True, alloc_graph_output: bool = True, alloc_mutable_buffers: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/configs.py#L168)

**Parameters:**

backend (`str`, *optional*, defaults to `"xnnpack"`) : Target ExecuTorch backend. Supported values:  - `"xnnpack"` — CPU inference via the XNNPACK library (default; runs anywhere). - `"cuda"` — GPU inference via the ExecuTorch CUDA backend. - `"mlx"` — GPU inference via the ExecuTorch MLX backend on Apple Silicon.

alloc_graph_input (`bool`, *optional*, defaults to `True`) : Whether the memory-planning pass reserves arena memory for graph inputs. When `False`, the runtime uses the caller-provided input buffers directly instead of copying into the arena — so an in-place `USER_INPUT_MUTATION` (e.g. a `StaticCache` write) lands in the caller's tensor rather than an arena copy.

alloc_graph_output (`bool`, *optional*, defaults to `True`) : Whether the memory-planning pass reserves arena memory for graph outputs. When `False`, the caller must bind output buffers at runtime (`Method::set_output_data_ptr`); binding an output to its mutated input's buffer avoids the copy-out roundtrip.

alloc_mutable_buffers (`bool`, *optional*, defaults to `True`) : Whether the memory-planning pass reserves arena memory for mutable buffers (model-resident state). Passed through to the `MemoryPlanningPass`.

Configuration class for exporting models to ExecuTorch format.

Inherits all fields from `DynamoConfig` (`dynamic`, `strict`,
`dynamic_shapes`, `prefer_deferred_runtime_asserts_over_guards`).

## OpenVINOConfig[[transformers.exporters.OpenVINOConfig]]

#### transformers.exporters.OpenVINOConfig[[transformers.exporters.OpenVINOConfig]]

```python
transformers.exporters.OpenVINOConfig(export_format: ExportFormat = <ExportFormat.OPENVINO: 'openvino'>, dynamic: bool = False, strict: bool = False, dynamic_shapes: dict[str, Any] | None = None, prefer_deferred_runtime_asserts_over_guards: bool = False, quantizer: ExportQuantizer | None = None, compress_to_fp16: bool = False, stateful: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/configs.py#L205)

**Parameters:**

compress_to_fp16 (*bool*, *optional*, defaults to *False*) : Compress the converted model's *float32* weights to *float16*, halving its size. Off by default for the narrower range.

stateful (*bool*, *optional*, defaults to *True*) : Fold round-tripped state tensors (KV cache, SSM states, …) into internal OV variables (`ReadValue`/``Assign`). The runtime then carries state across `infer()` calls instead of marshalling cache tensors through inputs/outputs on every step, and a fused `beam_idx`` input reorders state in-graph for beam search. No-op for models without round-tripped state (encoders, prefill-only exports). Set it to *False* for targets that take no stateful model, such as the NPU plugin.

Configuration class for exporting models to OpenVINO IR via `openvino.convert_model`.

Inherits all fields from [*DynamoConfig*] (*dynamic*, *strict*, *dynamic_shapes*,
*prefer_deferred_runtime_asserts_over_guards*).

## Quantizers[[transformers.exporters.ExportQuantizer]]

#### transformers.exporters.ExportQuantizer[[transformers.exporters.ExportQuantizer]]

```python
transformers.exporters.ExportQuantizer(calibration_dataset: Iterable[dict[str, Any]] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/base.py#L33)

**Parameters:**

calibration_dataset (`Iterable[dict]`, *optional*) : Input dicts the quantizer calibrates on: a list, or a `DataLoader` whose batches collate to model inputs. For `~HfExporter.export_for_generation` they are generate kwargs, captured into a calibration set per component. When `None`, a quantizer that needs data calibrates on the export's own sample inputs, with a warning.

Base class for post-training quantizers applied during export.

A quantizer declares the export formats it supports and the packages it needs (with the versions it is tested on,
like the exporters) and whether it quantizes the `torch.export` FX graph, before any conversion (so every backend
takes the result), or the model a backend converts it into (`onnx_ir.Model`, `openvino.Model`). The exporter
validates it against the export up front, then calls `~ExportQuantizer.quantize` with that model and the export's
sample inputs. Subclass it and implement `quantize` to add a new quantization method; a `CalibrationSet` maps the
calibration samples to the model's own inputs.

#### quantize[[transformers.exporters.ExportQuantizer.quantize]]

```python
quantize(model: Any, sample_inputs: dict[str, Any], export_format: ExportFormat)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/base.py#L83)

Quantize `model` (the FX graph if `quantizes_fx_graph`, else the backend model) and return the result,
calibrated on `calibration_dataset`, or on the export's `sample_inputs` when it has none.

#### traced_calibration[[transformers.exporters.ExportQuantizer.traced_calibration]]

```python
traced_calibration(sample_inputs: dict[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/base.py#L76)

Calibration samples as the traced kwargs, for FX-graph quantizers.

#### validate_environment[[transformers.exporters.ExportQuantizer.validate_environment]]

```python
validate_environment(export_format: ExportFormat)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/base.py#L68)

Raise if this quantizer can't run on an `export_format` export, or a package it needs is missing or too old;
warn when one differs from the version it is tested on.

#### with_calibration[[transformers.exporters.ExportQuantizer.with_calibration]]

```python
with_calibration(calibration_dataset: Iterable[dict[str, Any]])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/base.py#L62)

A copy of this quantizer that calibrates on `calibration_dataset`.

#### transformers.exporters.quantizers.CalibrationSet[[transformers.exporters.quantizers.CalibrationSet]]

```python
transformers.exporters.quantizers.CalibrationSet(calibration_dataset: Iterable[dict[str, Any]] | None, sample_inputs: dict[str, Any], transform: Callable[[dict[str, Any]], Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/base.py#L89)

Re-iterable view of calibration samples as model inputs.

The samples are the quantizer's `calibration_dataset`, or the export's own `sample_inputs` (with a warning, once
they are read) when it has none. They can be any mapping, such as a tokenizer's `BatchEncoding`. `transform` maps
each forward-kwarg sample to what the quantized model takes (traced kwargs, ONNX feeds, OpenVINO
port feeds), lazily, one sample at a time, on every pass.

#### transformers.exporters.PT2EQuantizer[[transformers.exporters.PT2EQuantizer]]

```python
transformers.exporters.PT2EQuantizer(quantizer: Any, calibration_dataset: Iterable[dict[str, Any]] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_pt2e.py#L33)

**Parameters:**

quantizer (`torchao.quantization.pt2e.quantizer.Quantizer`) : The PT2E quantizer that annotates the graph. Its quantize/dequantize ops must be ones the target backend supports: `X86InductorQuantizer` for inductor, ONNX (QDQ) or OpenVINO, `XNNPACKQuantizer` for ExecuTorch.

calibration_dataset (`Iterable[dict]`, *optional*) : Input dicts the observers calibrate on (see `ExportQuantizer`), one forward each. To calibrate on batches, pass a `DataLoader` that collates the samples into batched input dicts instead of a list of single samples.

Quantize the FX graph with [PyTorch 2 Export (PT2E) quantization](https://docs.pytorch.org/ao/main/pt2e_quantization/index.html):
`prepare_pt2e` inserts observers, the calibration set drives them, and `convert_pt2e` replaces them with
quantize/dequantize ops. Works with every export backend.

Example:

```python
>>> from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
...     X86InductorQuantizer,
...     get_default_x86_inductor_quantization_config,
... )
>>> from transformers.exporters import DynamoConfig, PT2EQuantizer

>>> x86 = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config())
>>> DynamoConfig(quantizer=PT2EQuantizer(x86, calibration_dataset=samples))
```

#### transformers.exporters.OnnxRuntimeQuantizer[[transformers.exporters.OnnxRuntimeQuantizer]]

```python
transformers.exporters.OnnxRuntimeQuantizer(config: OnnxRuntimeStaticConfig | OnnxRuntimeDynamicConfig | None = None, calibration_dataset: Iterable[dict[str, Any]] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_onnxruntime.py#L78)

Quantize the converted ONNX model with ONNX Runtime's own tools (`onnxruntime.quantization`): `quantize_static`
with an `OnnxRuntimeStaticConfig` (the default), or `quantize_dynamic` with an `OnnxRuntimeDynamicConfig`.

Example:

```python
>>> from transformers.exporters import (
...     OnnxConfig,
...     OnnxRuntimeDynamicConfig,
...     OnnxRuntimeQuantizer,
...     OnnxRuntimeStaticConfig,
... )

>>> static = OnnxRuntimeStaticConfig(per_channel=True)
>>> OnnxConfig(quantizer=OnnxRuntimeQuantizer(static, calibration_dataset=samples))
>>> OnnxConfig(quantizer=OnnxRuntimeQuantizer(OnnxRuntimeDynamicConfig()))
```

#### transformers.exporters.OnnxRuntimeStaticConfig[[transformers.exporters.OnnxRuntimeStaticConfig]]

```python
transformers.exporters.OnnxRuntimeStaticConfig(quant_format: Any = None, op_types_to_quantize: list[str] | None = None, per_channel: bool | None = None, reduce_range: bool | None = None, activation_type: Any = None, weight_type: Any = None, nodes_to_quantize: list[str] | None = None, nodes_to_exclude: list[str] | None = None, calibrate_method: Any = None, calibration_providers: list[str] | None = None, extra_options: dict[str, Any] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_onnxruntime.py#L41)

Arguments of `onnxruntime.quantization.quantize_static`: int8 activations and weights (QDQ by default), calibrated
on the quantizer's `calibration_dataset`. A field left to `None` keeps ONNX Runtime's default. See
[`quantize_static`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) for each one.

#### transformers.exporters.OnnxRuntimeDynamicConfig[[transformers.exporters.OnnxRuntimeDynamicConfig]]

```python
transformers.exporters.OnnxRuntimeDynamicConfig(op_types_to_quantize: list[str] | None = None, per_channel: bool | None = None, reduce_range: bool | None = None, weight_type: Any = None, nodes_to_quantize: list[str] | None = None, nodes_to_exclude: list[str] | None = None, extra_options: dict[str, Any] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_onnxruntime.py#L62)

Arguments of `onnxruntime.quantization.quantize_dynamic`: int8 weights, activations quantized at runtime, no
calibration. A field left to `None` keeps ONNX Runtime's default. See
[`quantize_dynamic`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) for each one.

#### transformers.exporters.NNCFOpenVINOQuantizer[[transformers.exporters.NNCFOpenVINOQuantizer]]

```python
transformers.exporters.NNCFOpenVINOQuantizer(config: NNCFQuantizationConfig | NNCFWeightCompressionConfig | None = None, calibration_dataset: Iterable[dict[str, Any]] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_nncf.py#L143)

Quantize the converted OpenVINO model with [NNCF](https://github.com/openvinotoolkit/nncf): `nncf.quantize` with a
`NNCFQuantizationConfig` (the default), or `nncf.compress_weights` with a `NNCFWeightCompressionConfig`.

Example:

```python
>>> import nncf
>>> from transformers.exporters import NNCFOpenVINOQuantizer, NNCFWeightCompressionConfig, OpenVINOConfig

>>> OpenVINOConfig(quantizer=NNCFOpenVINOQuantizer(calibration_dataset=samples))
>>> OpenVINOConfig(
...     quantizer=NNCFOpenVINOQuantizer(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT4_SYM))
... )
```

#### transformers.exporters.NNCFOnnxQuantizer[[transformers.exporters.NNCFOnnxQuantizer]]

```python
transformers.exporters.NNCFOnnxQuantizer(config: NNCFQuantizationConfig | NNCFWeightCompressionConfig | None = None, calibration_dataset: Iterable[dict[str, Any]] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_nncf.py#L175)

Quantize the converted ONNX model with [NNCF](https://github.com/openvinotoolkit/nncf) through its ONNX backend:
`nncf.quantize` with a `NNCFQuantizationConfig` (the default, int8 QDQ), or `nncf.compress_weights` with a
`NNCFWeightCompressionConfig` (int4 as ONNX Runtime's `MatMulNBits`).

Example:

```python
>>> import nncf
>>> from transformers.exporters import NNCFOnnxQuantizer, NNCFWeightCompressionConfig, OnnxConfig

>>> OnnxConfig(quantizer=NNCFOnnxQuantizer(calibration_dataset=samples))
>>> OnnxConfig(quantizer=NNCFOnnxQuantizer(NNCFWeightCompressionConfig(mode=nncf.CompressWeightsMode.INT4_SYM)))
```

#### transformers.exporters.NNCFTorchFXQuantizer[[transformers.exporters.NNCFTorchFXQuantizer]]

```python
transformers.exporters.NNCFTorchFXQuantizer(config: NNCFQuantizationConfig | NNCFWeightCompressionConfig | None = None, calibration_dataset: Iterable[dict[str, Any]] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_nncf.py#L231)

Quantize the `torch.export` FX graph with [NNCF](https://github.com/openvinotoolkit/nncf) through its TorchFX
backend: `nncf.quantize` with a `NNCFQuantizationConfig` (the default, `quantize`/`dequantize` ops), or
`nncf.compress_weights` with a `NNCFWeightCompressionConfig`. Like `PT2EQuantizer`, it runs before any
conversion, so every backend takes its result. ExecuTorch lowers only the weight-only result: XNNPACK doesn't
delegate the `quantize`/`dequantize` ops, and ExecuTorch has no kernels for them.

Example:

```python
>>> from transformers.exporters import DynamoConfig, NNCFTorchFXQuantizer

>>> DynamoConfig(quantizer=NNCFTorchFXQuantizer(calibration_dataset=samples))
```

#### transformers.exporters.NNCFQuantizationConfig[[transformers.exporters.NNCFQuantizationConfig]]

```python
transformers.exporters.NNCFQuantizationConfig(mode: Any = None, preset: Any = None, target_device: Any = None, subset_size: int | None = None, fast_bias_correction: bool | None = None, model_type: Any = None, ignored_scope: Any = None, advanced_parameters: Any = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_nncf.py#L58)

Arguments of `nncf.quantize`: int8 activations and weights, calibrated on the quantizer's `calibration_dataset`.
A field left to `None` keeps NNCF's default, except `model_type`, which defaults to `nncf.ModelType.TRANSFORMER`.
See [`nncf.quantize`](https://openvinotoolkit.github.io/nncf/autoapi/nncf/index.html#nncf.quantize) for each one.

#### transformers.exporters.NNCFWeightCompressionConfig[[transformers.exporters.NNCFWeightCompressionConfig]]

```python
transformers.exporters.NNCFWeightCompressionConfig(mode: Any = None, ratio: float | None = None, group_size: int | None = None, ignored_scope: Any = None, all_layers: bool | None = None, sensitivity_metric: Any = None, subset_size: int | None = None, awq: bool | None = None, scale_estimation: bool | None = None, gptq: bool | None = None, lora_correction: bool | None = None, backup_mode: Any = None, compression_format: Any = None, advanced_parameters: Any = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/quantizers/quantizer_nncf.py#L76)

Arguments of `nncf.compress_weights`: weight-only int8 or int4, with data-aware methods (AWQ, scale estimation,
GPTQ) for the int4 modes. The int8 modes are data-free; the others calibrate on the quantizer's
`calibration_dataset`. `gptq` and `lora_correction` are OpenVINO-only. A field left to `None` keeps NNCF's
default. See
[`nncf.compress_weights`](https://openvinotoolkit.github.io/nncf/autoapi/nncf/index.html#nncf.compress_weights) for
each one.

## Utilities[[transformers.exporters.utils.get_leaf_tensors]]

Lower-level functions that power `export_for_generation`, useful when you need to intervene
between decomposing a model and exporting each component.

#### transformers.exporters.utils.get_leaf_tensors[[transformers.exporters.utils.get_leaf_tensors]]

```python
transformers.exporters.utils.get_leaf_tensors(obj: Any)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/utils.py#L348)

**Parameters:**

obj (`Any`) : A tensor, dataclass, dict, list, tuple, or any nesting thereof.

**Returns:** `dict[str, torch.Tensor]`

Flat mapping from dotted path strings to tensors.

Recursively retrieve all leaf tensors from a potentially nested structure.

#### transformers.exporters.utils.prepare_for_export[[transformers.exporters.utils.prepare_for_export]]

```python
transformers.exporters.utils.prepare_for_export(model: PreTrainedModel | torch.nn.Module, inputs: MutableMapping[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/utils.py#L519)

Configure model and inputs for export. Mutates both `model` and `inputs` in place,
returning `(model, inputs, output_flags)` where `output_flags` holds the values popped
from `inputs` for `use_cache`, `return_dict`, etc. (to be applied reversibly onto
`model.config` by `patch_model_config` during the trace).

- Strips label inputs (`labels`, `future_values`) — loss computation is unsupported.
- Pops output flags (`use_cache`, `return_dict`, …) from `inputs` so they don't appear
  as traced kwargs; the values are returned for the trace block to apply onto
  `model.config`.
- Pre-computes data-dependent vision/audio kwargs registered via
  `@register_export_input_preparer` and writes them into `inputs`.
- Casts input tensors to match the model's `dtype` / `device`.

#### transformers.exporters.utils.decompose_prefill_decode[[transformers.exporters.utils.decompose_prefill_decode]]

```python
transformers.exporters.utils.decompose_prefill_decode(model: PreTrainedModel, inputs: dict[str, Any], generation_config: Any = None, multi_token_decode: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/utils.py#L922)

**Returns:** `dict[str, tuple[torch.nn.Module, dict]]`

`{"prefill": (model, prefill_inputs), "decode": (model, decode_inputs)}`.

Run `model.generate()` and capture prefill and decode inputs.

Reuses the full generation machinery so every architecture (decoder-only, SSM,
encoder-decoder, multi-modal, …) gets correct inputs without reimplementing the loop.

`generation_config` is forwarded to `generate()` (defaulting to the model's own), so the captured
inputs use whatever cache `generate()` would build. Pass one with `cache_implementation="static"`
and `max_cache_len=N` to capture a **statically sized** cache in the decode inputs — the basis for
a static-cache export. `max_cache_len` sizes the cache independently of the capture, so the
exported decode takes a fixed `[..., N, ...]` cache rather than a growing one.

When `multi_token_decode`, the `decode` component is captured as a **multi-token** decode — two
consecutive decode steps merged (see `_merge_decode_calls`) so its query-sequence axis stays
symbolic (a single-token decode would specialize that axis to 1). It then handles both one token
(ordinary decoding) and many (continuation-from-past, or a plain prefill when the cache is empty). Otherwise `decode` is the
classic single-token decode.

#### transformers.exporters.utils.decompose_multimodal[[transformers.exporters.utils.decompose_multimodal]]

```python
transformers.exporters.utils.decompose_multimodal(model: PreTrainedModel, inputs: dict[str, Any])
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/utils.py#L1045)

**Returns:** `dict[str, tuple[torch.nn.Module, dict]]`

One `name: (module, inputs)`
entry per detected submodule (image/audio encoder, projector, language model, lm_head).

**Raises:** ``ValueError``

- ``ValueError`` -- if no known multi-modal submodules are found on the model.

Capture inputs to each multi-modal submodule via a single forward pass.

Detects all known multi-modal submodules by attribute name (vision tower, projector,
language model, lm_head, …) and captures their forward kwargs during one
`model(**inputs)` call.

Each submodule is returned as a separate `name: (module, inputs)` entry for
independent export. The token-merge step (e.g. `masked_scatter` for multi-modal models)
is intentionally left outside the exported graphs — it is the caller's responsibility
to assemble `inputs_embeds` from the encoder outputs before running the decoder.

#### transformers.exporters.utils.decompose_for_generation[[transformers.exporters.utils.decompose_for_generation]]

```python
transformers.exporters.utils.decompose_for_generation(model: PreTrainedModel, inputs: dict[str, Any], generation_config: Any = None, multi_token_decode: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/utils.py#L1089)

**Parameters:**

model : Generative model. Must support `model.generate(**inputs)`.

inputs : **Generate** kwargs — what you'd pass to `model.generate(**inputs)`.

generation_config : Optional `GenerationConfig` forwarded to `generate()` during capture. Pass one with `cache_implementation="static"` + `max_cache_len=N` to export against a statically sized cache (see `decompose_prefill_decode`).

multi_token_decode : When `True`, capture the `decode` component as a multi-token decode (dynamic query sequence axis: multiple tokens at once — continuation-from-past, or a plain prefill when the cache is empty); a single-token decode can't stay dynamic (see `decompose_prefill_decode`).

**Returns:** `{component_name

(submodel, forward_inputs)}`. Keys are `"prefill"` / `"decode"` for
plain generative models and `"<modality>_encoder"` / `"multi_modal_projector"` /
`"language_model"` / `"lm_head"` / `"decode"` for multi-modal generative models.

Decompose a generative model into independently exportable `(model, forward_inputs)` pairs.

Runs `decompose_prefill_decode` to capture prefill and decode forward kwargs from a real
`model.generate(**inputs, max_new_tokens=2)`. If the prefill is multi-modal (per `is_multimodal`),
further splits it into one entry per submodule (vision/audio encoder, projector, language model,
`lm_head`) via `decompose_multimodal`.

#### transformers.exporters.utils.is_multimodal[[transformers.exporters.utils.is_multimodal]]

```python
transformers.exporters.utils.is_multimodal(model: PreTrainedModel | torch.nn.Module)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/exporters/utils.py#L1036)

Returns `True` if the model is multi-modal with modal encoders and a language model.

A non-`PreTrainedModel` (e.g. a bare `nn.Module`) has no canonical `get_encoder`/`get_decoder`
accessors and is trivially not multi-modal, so it short-circuits to `False`.

### Callbacks
https://huggingface.co/docs/transformers/v5.23.0/main_classes/callback.md
