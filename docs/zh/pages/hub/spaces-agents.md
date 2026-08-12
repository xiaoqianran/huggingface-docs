<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间作为代理工具

每个 Gradio Space 都公开一个纯文本 `agents.md`，编码代理（Claude Code、Codex、OpenCode、Pi 等）可以直接调用。通过 [huggingface.co/spaces](https://huggingface.co/spaces) 上的语义搜索（例如“音频转录”）找到一个，可以选择先在 UI 中尝试它，然后将代理指向其 `agents.md`。响应一次性提供调用 Space 所需的一切：架构 URL、调用和轮询模板、文件上传说明以及身份验证提示。

当**链接空间**时，这会变得更加强大。代理可以通过调用图像的[⟦T7⟧](https://huggingface.co/spaces/black-forest-labs/flux-klein-9b-kv)，然后将生成的图像传递到3D模型的[⟦T8⟧](https://huggingface.co/spaces/microsoft/TRELLIS.2)，将提示转换为3D资产。没有客户端库，没有硬编码集成。

您只需要在您的环境中设置 [HF_TOKEN](https://huggingface.co/settings/tokens) 即可开始。

## 从用户界面

每个兼容的空间页面的标题中都有一个 **代理** 按钮。单击它可复制该空间的 `agents.md` 的 `curl` 命令，然后将其粘贴到您的代理中。

## Agents.md 端点

```
https://huggingface.co/spaces/<namespace>/<repo>/agents.md
```

示例：

```bash
curl https://huggingface.co/spaces/microsoft/TRELLIS.2/agents.md
```

返回：

```
To use this application (microsoft/TRELLIS.2: Create 3D model from a single image):
API schema: GET https://microsoft-trellis-2.hf.space/gradio_api/info
Call endpoint: POST https://microsoft-trellis-2.hf.space/gradio_api/call/v2/{endpoint} {"param_name": value, ...}
Poll result: GET https://microsoft-trellis-2.hf.space/gradio_api/call/{endpoint}/{event_id}
File inputs: POST https://microsoft-trellis-2.hf.space/gradio_api/upload -F "files=@file.ext", use as: {"path": "<returned-path>", "meta": {"_type": "gradio.FileData"}, "orig_name": "file.ext"}
Auth: Bearer $HF_TOKEN (https://huggingface.co/settings/tokens)
```

## 文件输入

当端点获取文件（图像、音频、视频等）时，请先上传该文件并在调用中引用返回的路径。上传到`/gradio_api/upload`并重用它返回的路径：

```bash
# 1. Upload the file
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://microsoft-trellis-2.hf.space/gradio_api/upload \
  -F "files=@chair.png"
# → ["/tmp/gradio/.../chair.png"]

# 2. Reference the returned path as a FileData object in your call
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://microsoft-trellis-2.hf.space/gradio_api/call/v2/predict \
  -d '{"image": {"path": "/tmp/gradio/.../chair.png", "meta": {"_type": "gradio.FileData"}, "orig_name": "chair.png"}}'
```接受公共 URL（而不是上传的文件）的输入可以直接传递，无需上传步骤。

## 身份验证和 ZeroGPU

大多数流行的空间都在 [ZeroGPU](./spaces-zerogpu) 上运行，它使用呼叫者的每日配额。代理应始终传递 `$HF_TOKEN`，以便通话费用计入您的帐户而不是受限制的匿名池。私人空间也需要相同的令牌。在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 创建一个（`agents.md` 授权行直接链接到此页面）。

```bash
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://microsoft-trellis-2.hf.space/gradio_api/call/v2/predict \
  -d '{"image": {"path": "https://example.com/chair.png"}}'
```

## 代理将如何使用它

1. 空间代理`curl`s `/agents.md`。
2. 它获取`/gradio_api/info`以了解端点名称和输入。
3. 对于文件输入，它将文件 POST 到`/gradio_api/upload` 并保留返回的路径。
4. 它 POST 到 `/gradio_api/call/v2/<endpoint>`，然后获取轮询 URL 以传输结果。

### 空间上的 JupyterLab
https://huggingface.co/docs/hub/spaces-sdks-docker-jupyter.md