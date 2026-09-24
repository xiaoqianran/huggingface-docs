<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 迁移到huggingface_hub v2.0

`huggingface_hub` v2.0 将其 HTTP 依赖项替换为 [⟦T2⟧](https://httpx2.pydantic.dev/)，并删除了已弃用的 1.x API。 HTTP 客户端和异常现在来自单独的包。

要同时支持 `huggingface_hub` v1.x（从 v1.30.0 开始）和 v2.x，请从 `huggingface_hub.utils` 导入 HTTP 模块：

```python
from huggingface_hub.utils import get_session, httpx

try:
    response = get_session().get("https://huggingface.co/api/models/gpt2")
    response.raise_for_status()
except httpx.HTTPError:
    ...
```

为 [set_client_factory()](/docs/huggingface_hub/v2.0.0/en/package_reference/utilities#huggingface_hub.set_client_factory) 或 [set_async_client_factory()](/docs/huggingface_hub/v2.0.0/en/package_reference/utilities#huggingface_hub.set_async_client_factory) 创建自定义客户端时，请使用相同的导入。旧 HTTP 包中的客户端和异常与 `httpx2` 中的客户端和异常不同，不能互换使用。

`httpx2` 默认使用操作系统的证书信任存储。仍然支持使用 `SSL_CERT_FILE` 或 `SSL_CERT_DIR` 配置的自定义 CA 捆绑包。日志记录配置应针对 `httpx2` 和 `httpcore2` 而不是旧的记录器名称。详情请参阅[upstream migration guide](https://httpx2.pydantic.dev/migration/)。

`oauth` extra 现在需要 `authlib>=1.8.0` 来支持 `httpx2`。

## 删除了已弃用的 API- 使用`upload_folder`代替`upload_large_folder`或`HfApi.upload_large_folder`。
- 使用 `duplicate_repo`、`set_space_volumes` 和 `delete_space_volumes` 代替 `duplicate_space`、`request_space_storage` 和 `delete_space_storage`。
- 使用`parse_hf_uri`代替`repo_type_and_id_from_hf_id`，使用`list_models(search=...)`代替`model_name=...`。
- 在`create_repo`和`duplicate_repo`中使用`space_volumes`代替`space_storage`。
- 使用`InferenceEndpointType.AUTHENTICATED`或`type="authenticated"`代替`PROTECTED`或`type="protected"`。
- 使用`text_generation(stop=...)`代替`stop_sequences=...`，并将令牌字符串或`None`传递给`InferenceClient`而不是布尔值。
- 使用`hf`代替`huggingface-cli`； `hf repo` 和 `hf repo-files delete` 替换为 `hf repos` 和 `hf repos delete-files`。
- 使用`hf upload`代替`hf upload-large-folder`；使用`--volume`代替`--storage`进行存储库创建和复制，并使用`--status`或`--label`代替`hf jobs ps --filter`。
- `hf skills add` 和 `hf skills update` 上已弃用的 `--claude` 选项已删除。克劳德代码的技能会自动安装。

有关更多上下文，请参阅 [⟦T55⟧ project](https://github.com/pydantic/httpx2)，
[⟦T56⟧ documentation](https://pydantic.dev/docs/httpx2/)，上游
[transition guide from ⟦T57⟧ to ⟦T58⟧](https://httpx2.pydantic.dev/migration/)，以及
[⟦T59⟧ transition plan](https://github.com/huggingface/huggingface_hub/issues/4802)。

### 引擎盖下的沙箱
https://huggingface.co/docs/huggingface_hub/v2.0.0/concepts/sandbox.md