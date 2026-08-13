<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型下载统计数据

## 模型的下载量是如何计算的？

计算模型的下载数量并不是一项简单的任务，因为单个模型存储库可能包含多个文件，包括多个模型权重文件（例如，具有分片模型）和不同的格式，具体取决于库（GGUF、PyTorch、TensorFlow 等）。为了避免重复计算下载（例如，将模型的单次下载计算为多次下载），集线器使用一组用于下载计数的查询文件。用户不会发送任何信息，也不会为此进行任何其他调用。计数是在服务器端完成的，因为集线器提供文件下载服务。

对这些文件的每个 HTTP 请求（包括 `GET` 和 `HEAD`）都将计为下载。默认情况下，当未指定库时，Hub 使用 `config.json` 作为默认查询文件。否则，查询文件取决于每个库，并且 Hub 可能会检查 `pytorch_model.bin` 或 `adapter_config.json` 等文件。 

## 不同库的查询文件有哪些？默认情况下，Hub 会查看 `config.json`、`config.yaml`、`hyperparams.yaml`、`params.json` 和 `meta.yaml`。一些库通过指定自己的过滤器（指定`countDownloads`）来覆盖这些默认值。定义这些覆盖的代码是[open-source](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts)。例如，对于`nemo`库，所有具有`.nemo`扩展名的文件都用于计算下载量。

## 我可以为我的图书馆添加查询文件吗？ 

是的，您可以打开拉取请求[here](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts)。这是为 VFIMamba 添加下载指标的最小 [example](https://github.com/huggingface/huggingface.js/pull/885/files)。查看[integration guide](./models-adding-libraries#register-your-library)了解更多详情。

## `GGUF` 文件如何处理？

GGUF 文件是独立的，不依赖于单个库，因此所有这些文件都会计入下载量。如果用户执行整个存储库的克隆，这将双重计算下载，但大多数用户和界面都会为给定存储库下载单个 GGUF 文件。

## `diffusers` 是如何处理的？

`diffusers` 库是一种边缘情况，其过滤器在内部代码库中配置。该过滤器确保标记为 `diffusers` 的存储库对通过库以及通过需要用户手动下载顶级安全张量的 UI 加载的文件进行计数。

```
filter: [
		{
			bool: {
				/// Include documents that match at least one of the following rules
				should: [
					/// Downloaded from diffusers lib
					{
						term: { path: "model_index.json" },
					},
					/// Downloaded from diffusers lib through modular pipelines
					{
						term: { path: "modular_model_index.json" },
					},
					/// Direct downloads (LoRa, Auto1111 and others)
					/// Filter out nested safetensors and pickle weights to avoid double counting downloads from the diffusers lib
					{
						regexp: { path: "[^/]*\\.safetensors" },
					},
					{
						regexp: { path: "[^/]*\\.ckpt" },
					},
					{
						regexp: { path: "[^/]*\\.bin" },
					},
				],
				minimum_should_match: 1,
			},
		},
	]
}
```

## 如果我的模型需要更精细的下载数据怎么办？如果您需要更精细的下载数据，例如：
- 区分`config.json`和模型权重，
- 排除来自 CI/CD 管道的下载，
- 或删除重复用户（即计算唯一的下载者），

那么[Publisher Analytics](./publisher-analytics)，特别是[granular logs](./publisher-analytics#unique-downloaders-and-more-granular-logs)功能，可以为您的组织发布的所有模型和数据集提供匿名的请求级访问日志。

这些作为原始日志提供，因为大多数组织都希望应用自己的自定义规则。

### 许可证
https://huggingface.co/docs/hub/repositories-licenses.md