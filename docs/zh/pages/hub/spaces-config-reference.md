<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间配置参考

空间是通过存储库根目录的 **README.md** 文件顶部的 `YAML` 块配置的。下面列出了所有可接受的参数。

**`title`** : _string_  
显示空间的标题。  

**`emoji`** : _string_  
空格表情符号（允许仅使用表情符号字符）。  

**`colorFrom`**：_string_  
缩略图渐变的颜色（红色、黄色、绿色、蓝色、靛蓝、紫色、粉色、灰色）。  

**`colorTo`** : _string_  
缩略图渐变的颜色（红色、黄色、绿色、蓝色、靛蓝、紫色、粉色、灰色）。  

**`sdk`** : _string_  
可以是 `gradio`、`docker` 或 `static`。  

**`python_version`**：_string_  
任何有效的 Python `3.x` 或 `3.x.x` 版本。  
默认为`3.10`。  

**`sdk_version`**：_string_  
指定要使用的 Gradio 版本。
支持所有版本的 Gradio。**`suggested_hardware`** : _string_  
指定必须在其上运行该空间的建议 [hardware](https://huggingface.co/docs/hub/spaces-gpus)。  
对于要由其他用户复制的空间很有用。  
设置此值不会自动将硬件分配给此空间。  
值必须是有效的硬件风格。当前有效的硬件风格：
- CPU：`"cpu-basic"`、`"cpu-upgrade"`
- GPU：`"t4-small"`、`"t4-medium"`、`"l4x1"`、`"l4x4"`、
	`"l40sx1"`、`"l40sx4"`、`"l40sx8"`、
	`"a10g-small"`、`"a10g-large"`、`"a10g-largex2"`、
	`"a10g-largex4"`、`"a100-large"`、`"a100x4"`、`"a100x8"`

**`suggested_storage`** : _string_  
指定必须在其上运行该空间的建议 [permanent storage](https://huggingface.co/docs/hub/spaces-storage)。  
对于要由其他用户复制的空间很有用。  
设置此值不会自动为该空间分配永久存储。  
值必须是 `"small"`、`"medium"` 或 `"large"` 之一。  
> [!注意]
> 持久存储功能不再可用，因此此设置将被忽略。

**`app_file`** : _string_  
主应用程序文件的路径（包含 `gradio` Python 代码或 `static` html 代码）。  
路径是相对于存储库的根目录的。  

**`app_build_command`**：_string_  
对于静态空间，首先运行命令以生成要渲染的 HTML。示例：`npm run build`。 

它与指向构建索引文件的`app_file`结合使用：例如`app_file: dist/index.html`。每次更新，构建命令都会在 Job 中运行，构建输出将存储在 `refs/convert/build` 中，
这将由空间提供服务。请参阅 https://huggingface.co/spaces/coyotte508/static-vite 的示例

**`app_port`** : _int_  
您的应用程序正在其上运行的端口。仅当 `sdk` 为 `docker` 时使用。默认端口是`7860`。

**`base_path`**：_string_
对于非静态空间，要渲染的初始 url。需要从`/`开始。对于静态空间，请使用 `app_file` 代替。

**`fullWidth`**：_布尔值_  
您的空间是在 iframe 内的全角（当`true`）还是固定宽度列（即“容器”CSS）内呈现。
默认为`true`。

**`header`**：_string_  
可以是 `mini` 或 `default`。如果`header`设置为`mini`，空间将全屏显示，并带有迷你浮动标题。   

**`short_description`**：_string_
空间的简短描述。这将显示在空间的缩略图中。

**`models`** : _列表[字符串]_  
空间中使用的 HF 型号 ID（例如 `openai-community/gpt2` 或 `deepset/roberta-base-squad2`）。
如果此处未指定，将从您的代码中自动解析。  

**`datasets`** : _列表[字符串]_  
空间中使用的 HF 数据集 ID（例如 `mozilla-foundation/common_voice_13_0` 或 `oscar-corpus/OSCAR-2109`）。
如果此处未指定，将从您的代码中自动解析。**`tags`** : _列表[字符串]_  
描述您的 Space 任务或范围的术语列表。  

**`thumbnail`**：_string_  
用于定义社交共享自定义缩略图的 URL。

**`pinned`** : _boolean_  
空间是否位于您的个人资料顶部。如果您有很多空间，这会很有用，这样您和其他人就可以快速看到您最好的空间。  

**`hf_oauth`** : _boolean_  
连接的 OAuth 应用程序是否与此空间关联。更多详情请参阅[Adding a Sign-In with HF button to your Space](https://huggingface.co/docs/hub/spaces-oauth)。

**`hf_oauth_scopes`** : _列表[字符串]_
连接的 OAuth 应用程序的授权范围。 `openid`、`profile` 默认授权，不需要此参数。更多详情请参见[Adding a Sign-In with HF button to your space](https://huggingface.co/docs/hub/spaces-oauth)。

**`hf_oauth_expiration_minutes`** : _int_
OAuth 令牌的持续时间（以分钟为单位）。默认为 480 分钟（8 小时）。最长持续时间为 43200 分钟（30 天）。更多详情请参见[Adding a Sign-In with HF button to your space](https://huggingface.co/docs/hub/spaces-oauth)。

**`hf_oauth_authorized_org`** : _string_ 或 _List[string]_
将 OAuth 访问限制为特定组织的成员。更多详情请参阅[Adding a Sign-In with HF button to your space](https://huggingface.co/docs/hub/spaces-oauth)。

**`disable_embedding`** : _boolean_
Space iframe是否可以嵌入其他网站。
默认为 false，即*可以*嵌入空格。**`startup_duration_timeout`**：_string_
为您的空间设置自定义启动持续时间超时。这是您的空间在超时并被标记为不健康之前允许启动的最长时间。
默认为 30 分钟，但任何有效的持续时间（如 `1h`、`30m`）都是可接受的。

**`custom_headers`** : _Dict[字符串, 字符串]_  
设置自定义 HTTP 标头，这些标头将在为您的空间提供服务时添加到所有 HTTP 响应中。  
目前，仅允许使用 [cross-origin-embedder-policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cross-Origin-Embedder-Policy) (COEP)、[cross-origin-opener-policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cross-Origin-Opener-Policy) (COOP) 和 [cross-origin-resource-policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cross-Origin-Resource-Policy) (CORP) 标头。这些标头可用于设置跨源隔离环境并启用`SharedArrayBuffer`等强大功能，例如：

```yaml
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
```

*注意：*所有标题和值都必须小写。

**`preload_from_hub`**：_列表[字符串]_
指定要在空间构建期间预加载的 Hugging Face Hub 模型或其他大文件的列表。这样可以在应用程序启动时准备好文件，从而优化启动时间。这对于依赖大型模型或数据集的空间特别有用，否则需要在运行时下载这些模型或数据集。每个项目的格式为 `"repository_name"` 用于从存储库下载所有文件，或 `"repository_name file1,file2"` 用于下载该存储库中的特定文件。您还可以使用格式 `"repository_name file1,file2 commit_sha256"` 指定要下载的特定提交。 

用法示例：
```yaml
preload_from_hub:
  - warp-ai/wuerstchen-prior text_encoder/model.safetensors,prior/diffusion_pytorch_model.safetensors
  - coqui/XTTS-v1
  - openai-community/gpt2 config.json 11c5a3d5811f50298f278a704980280950aedb10
```
在此示例中，Space 将在构建期间从 Hugging Face Hub 预加载来自 `warp-ai/wuerstchen-prior`、完整 `coqui/XTTS-v1` 存储库以及 `openai-community/gpt2` 存储库中 `config.json` 文件的特定修订版的特定 .safetensors 文件。

> [!警告]
> 文件保存在默认的`huggingface_hub`磁盘缓存`~/.cache/huggingface/hub`中。如果您的应用程序在其他地方需要它们或者您更改了 `HF_HOME` 变量，则此时不会遵循此预加载。

> [!注意]
> 尚不支持私有存储库的预加载。

### 上传数据集
https://huggingface.co/docs/hub/datasets-adding.md