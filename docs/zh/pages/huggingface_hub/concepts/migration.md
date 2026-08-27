<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 迁移到huggingface_hub v1.0

v1.0 版本是 `huggingface_hub` 库的一个重要里程碑。它标志着我们对 API 稳定性和库成熟度的承诺。我们进行了多项改进和重大更改，使该库更加强大且更易于使用。

本指南旨在帮助您将现有代码迁移到新版本。如果您有任何疑问或反馈，请通过[opening an issue on GitHub](https://github.com/huggingface/huggingface_hub/issues)告知我们。

## Python 3.9+

`huggingface_hub` 现在需要 Python 3.9 或更高版本。不再支持 Python 3.8。

## HTTPX 迁移

`huggingface_hub` 库现在使用 [⟦T3⟧](https://www.python-httpx.org/) 而不是 `requests` 来处理 HTTP 请求。进行此更改是为了提高性能并以相同的方式支持同步和异步请求。因此，我们删除了 `requests` 和 `aiohttp` 依赖项。

### 重大变更

这是影响整个图书馆的重大变化。虽然我们尝试使此更改尽可能透明，但在某些情况下您可能需要更新代码。以下是在此过程中引入的重大更改的列表：- **代理配置**：不再支持“按方法”代理。必须使用 `HTTP_PROXY` 和 `HTTPS_PROXY` 环境变量全局配置代理。
- **自定义 HTTP 后端**：`configure_http_backend` 功能已被删除。您现在应该使用 [set_client_factory()](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.set_client_factory) 和 [set_async_client_factory()](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.set_async_client_factory) 来配置 HTTP 客户端。
- **错误处理**：HTTP错误不再从`requests.HTTPError`继承，而是从`httpx.HTTPError`继承。我们建议捕获 `huggingface_hub.HfHubHttpError`，它是 v0.x 中 `requests.HTTPError` 和 v1.x 中 `httpx.HTTPError` 的子类。捕获 `huggingface_hub` 错误可确保您的代码与旧版本和新版本的库兼容。
- **SSLError**：`httpx`没有`SSLError`的概念。现在它是通用的`httpx.ConnectError`。
- **`LocalEntryNotFoundError`**：此错误不再继承自`HTTPError`。我们现在定义一个`EntryNotFoundError`（新），它由`LocalEntryNotFoundError`（如果在本地缓存中找不到文件）和`RemoteEntryNotFoundError`（如果在集线器上的存储库中找不到文件）继承。只有远程错误继承自`HTTPError`。
- **`InferenceClient`**：`InferenceClient`现在可以用作上下文管理器。当从语言模型流式传输令牌以确保连接正确关闭时，这特别有用。- **`AsyncInferenceClient`**：`trust_env` 参数已从 `AsyncInferenceClient` 的构造函数中删除。 `httpx` 默认信任环境变量。如果您明确不想信任该环境，则必须使用[set_client_factory()](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.set_client_factory)进行配置。

更多详情可以查看[PR #3328](https://github.com/huggingface/huggingface_hub/pull/3328)介绍`httpx`。

### 为什么`httpx`？

从 `requests` 到 `httpx` 的迁移带来了多项关键改进，增强了库的性能、可靠性和可维护性：

**线程安全和连接重用**：`httpx`在设计上是线程安全的，允许我们跨多个线程安全地重用同一个客户端。这种连接重用减少了为每个 HTTP 请求建立新连接的开销，从而提高了性能，尤其是在频繁向集线器发出请求时。

**HTTP/2 支持**：`httpx` 提供原生 HTTP/2 支持，这在向同一服务器发出多个请求时提供更高的效率（正是我们的用例）。与 HTTP/1.1 相比，这意味着更低的延迟和更少的资源消耗。**统一同步/异步 API**：与我们之前使用单独的 `requests`（同步）和 `aiohttp`（异步）依赖项的设置不同，`httpx` 为同步和异步客户端提供相同的行为。这确保了`InferenceClient`和`AsyncInferenceClient`具有一致的功能，并消除了两个实现之间先前存在的细微行为差异。

**改进 SSL 错误处理**：`httpx` 更优雅地处理 SSL 错误，使调试连接问题变得更容易、更可靠。

**面向未来的架构**：`httpx` 是为现代 Python 应用程序积极维护和设计的。相比之下，`requests` 处于维护模式，不会收到线程安全改进或 HTTP/2 支持等重大更新。

**更好的环境变量处理**：`httpx`在同步和异步上下文中提供更一致的环境变量处理，消除了以前的不一致性，即`requests`默认读取本地环境变量，而`aiohttp`不会。过渡到 `httpx` 使 `huggingface_hub` 具有现代、高效且可维护的 HTTP 后端。虽然大多数用户应该体验无缝操作，但底层改进为所有 Hub 交互提供了更好的性能和可靠性。

## `hf_transfer`

现在 Hub 上的所有存储库都启用了 Xet，并且 `hf_xet` 是下载/上传文件的默认方式，我们已经删除了对 `hf_transfer` 可选包的支持。因此，`HF_HUB_ENABLE_HF_TRANSFER` 环境变量被忽略。请改用[⟦T54⟧](../package_reference/environment_variables)。

## `Repository` 类

`Repository` 类已在 v1.0 中删除。它是 `git` CLI 的一个薄包装，用于管理存储库。您仍然可以直接在终端中使用`git`，但推荐的方法是使用`huggingface_hub`库中基于HTTP的API以获得更流畅的体验，特别是在处理大文件时。

以下是从旧 `Repository` 类到新 `HfApi` 类的映射：| `Repository` 方法 | `HfApi` 方法 |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| `repo.clone_from` | `snapshot_download` |
| `repo.git_add` + `git_commit` + `git_push` | [upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)、[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)、[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) |
| `repo.git_tag` | `create_tag` |
| `repo.git_branch` | `create_branch` |

## `HfFolder` 类

`HfFolder` 用于管理用户访问令牌。使用[login()](/docs/huggingface_hub/v1.29.0/en/package_reference/authentication#huggingface_hub.login)保存新令牌，使用[logout()](/docs/huggingface_hub/v1.29.0/en/package_reference/authentication#huggingface_hub.logout)删除它，使用[whoami()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.whoami)检查与当前令牌关联的用户。最后，使用 `get_token()` 在脚本中检索用户的令牌。

## `InferenceApi` 类

`InferenceApi` 是一个与 Inference API 交互的类。现在建议使用 [InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 类代替​​。

## 其他已弃用的功能

v1.0 中删除了一些方法和参数。下面列出的内容已在 v0.x 中被弃用，并带有警告消息。- `constants.hf_cache_home`已被删除。请使用`HF_HOME`代替。
- `use_auth_token` 参数已从所有方法中删除。请使用`token`代替。
- `get_token_permission` 方法已被删除。
- `update_repo_visibility` 方法已被删除。请使用`update_repo_settings`代替。
- `is_write_action` 参数已从`build_hf_headers` 中删除，`write_permission` 从`login` 中删除。 “写权限”的概念已被删除，并且不再相关，因为细粒度令牌是推荐的方法。
- 为了更加清晰，`login`中的`new_session`参数已重命名为`skip_if_logged_in`。
- `resume_download`、`force_filename`和`local_dir_use_symlinks`参数已从`hf_hub_download`和`snapshot_download`中删除。
- `library`、`language`、`tags` 和 `task` 参数已从 `list_models` 中删除。

## CLI 缓存命令

CLI 的缓存管理已重新设计，以遵循 Docker 启发的工作流程。已弃用的 `huggingface-cli` 已被删除，`hf`（在 v0.34 中引入）用更清晰的资源操作 CLI 取代它。
旧版 `hf cache scan` 和 `hf cache delete` 命令也在 v1.0 中删除，并替换为以下新的三个命令：- `hf cache ls` 使用简洁的表格、JSON 或 CSV 输出列出缓存条目。使用 `--revisions` 检查各个修订版本，添加 `--filter` 表达式，例如 `size>1GB` 或 `accessed>30d`，并在仅需要标识符时将它们与 `--quiet` 组合。
- `hf cache rm` 删除选定的缓存条目。传递一个或多个存储库 ID（例如 `model/bert-base-uncased`）或修订哈希值，并可选择添加 `--dry-run` 进行预览或 `--yes` 跳过确认提示。这取代了上一个命令中的交互式 TUI 和 `--disable-tui` 工作流程。
- `hf cache prune` 执行一次性删除未引用修订版的常见清理任务。以与`hf cache rm`相同的方式添加`--dry-run`或`--yes`。

最后，额外的 `[cli]` 已被删除 - CLI 现在附带核心 `huggingface_hub` 包。

## TensorFlow 和 Keras 2.x 支持

v1.0 中删除了所有与 TensorFlow 相关的代码和依赖项。这包括以下重大更改：

- `huggingface_hub[tensorflow]` 不再是受支持的额外依赖项
- `split_tf_state_dict_into_shards`和`get_tf_storage_size`实用功能已被删除。
- `tensorflow`、`fastai` 和 `fastcore` 版本不再包含在内置标头中。Keras 2.x 集成也已被删除。这包括 `KerasModelHubMixin` 类以及 `save_pretrained_keras`、`from_pretrained_keras` 和 `push_to_hub_keras` 实用程序。 Keras 2.x 是一个遗留且无人维护的库。推荐的方法是使用与 Hub 紧密集成的 Keras 3.x（即它包含加载/推送到 Hub 的内置方法）。如果您仍然想使用 Keras 2.x，您应该将 `huggingface_hub` 降级到 v0.x 版本。

## `upload_file` 和 `upload_folder` 返回值

[upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 和 [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 函数现在返回在 Hub 上创建的提交的 URL。以前，它们返回文件或文件夹的 URL。这是为了与 [create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)、[delete_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_file) 和 [delete_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_folder) 的返回值保持一致。