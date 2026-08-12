<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 你的第一个 Docker 空间：使用 T5 生成文本

在以下部分中，您将了解创建 Docker 空间、配置它以及将代码部署到其中的基础知识。我们将使用 Docker 创建一个 **文本生成** 空间，用于演示 [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) 模型，该模型可以使用 FastAPI 作为服务器，在给定一些输入文本的情况下生成文本。

您可以找到此托管[here](https://huggingface.co/spaces/DockerTemplates/fastapi_t5)的完整版本。

## 创建一个新的 Docker 空间

我们将从 [creating a brand new Space](https://huggingface.co/new-space) 开始并选择 **Docker** 作为我们的 SDK。

Hugging Face Spaces 是 Git 存储库，这意味着您可以通过推送提交来增量（协作）地处理您的空间。在继续之前，请查看 [Getting Started with Repositories](./repositories-getting-started) 指南，了解如何创建和编辑文件。如果您更喜欢使用 UI，您也可以直接在浏览器中完成工作。

当 [creating a new Space](https://huggingface.co/new-space) 时选择 **Docker** 作为 SDK，将通过在 `README.md` 文件的 YAML 块中将 `sdk` 属性设置为 `docker` 来初始化 Docker 空间。

```yaml
sdk: docker
```

您可以选择通过在 `README.md` 文件的 YAML 块中设置 `app_port` 属性来更改空间的默认应用程序端口。默认端口为`7860`。

```yaml
app_port: 7860
```

## 添加依赖对于 **文本生成** 空间，我们将构建一个 FastAPI 应用程序，展示名为 Flan T5 的文本生成模型。对于模型推理，我们将使用 [🤗 Transformers pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) 来使用模型。我们需要首先安装一些依赖项。这可以通过在我们的存储库中创建 **requirements.txt** 文件并向其添加以下依赖项来完成：

```
fastapi==0.74.*
requests==2.27.*
sentencepiece==0.1.*
torch==1.11.*
transformers==4.*
uvicorn[standard]==0.17.*
```

这些依赖项将安装在我们稍后创建的 Dockerfile 中。

## 创建应用程序

让我们使用一个虚拟的 FastAPI 应用程序来启动该过程，看看我们是否可以让端点正常工作。第一步是创建一个应用程序文件，在本例中，我们将其称为`main.py`。

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World!"}
```

## 创建 Dockerfile

Docker Space 的主要步骤是创建 Dockerfile。您可以阅读有关 Dockerfile [here](https://docs.docker.com/get-started/) 的更多信息。尽管我们在本教程中使用 FastAPI，但 Dockerfile 为用户提供了极大的灵活性，允许您构建新一代的 ML 演示。让我们为我们的应用程序编写 Dockerfile

```Dockerfile
# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

```

保存更改后，空间将重建，您的演示应该会在几秒钟后启动！ [Here](https://huggingface.co/spaces/DockerTemplates/fastapi_dummy) 是此时的一个示例结果。

### 本地测试**给高级用户的提示（您可以跳过）：** 如果您在本地开发，这是一个很好的时机，您可以执行`docker build`和`docker run`进行本地调试，但将更改推送到 Hub 并查看它的样子甚至更容易！

```bash
docker build -t fastapi .
docker run  -it -p 7860:7860 fastapi
```

如果您有 [Secrets](spaces-sdks-docker#secret-management)，您可以使用 `docker buildx` 并将秘密作为构建参数传递

```bash
export SECRET_EXAMPLE="my_secret_value"
docker buildx build --secret id=SECRET_EXAMPLE,env=SECRET_EXAMPLE -t fastapi .
```

并使用 `docker run` 将秘密作为环境变量传递

```bash
export SECRET_EXAMPLE="my_secret_value"
docker run -it -p 7860:7860 -e SECRET_EXAMPLE=$SECRET_EXAMPLE fastapi
```

## 添加一些机器学习到我们的应用程序

如前所述，我们的想法是使用 Flan T5 模型进行文本生成。我们想要为输入字段添加一些 HTML 和 CSS，因此让我们创建一个名为 static 的目录，其中包含 `index.html`、`style.css` 和 `script.js` 文件。此时，你的文件结构应该如下所示：

```bash
/static
/static/index.html
/static/script.js
/static/style.css
Dockerfile
main.py
README.md
requirements.txt
```

让我们完成所有步骤以使其正常工作。我们将跳过 CSS 和 HTML 的一些细节。您可以在[DockerTemplates/fastapi_t5](https://huggingface.co/spaces/DockerTemplates/fastapi_t5)空间的文件和版本选项卡中找到完整代码。

1.编写FastAPI端点进行推理

我们将使用 `transformers` 中的 `pipeline` 来加载 [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) 模型。我们将设置一个名为 `infer_t5` 的端点，用于接收、输入和输出推理调用的结果

```python
from transformers import pipeline

pipe_flan = pipeline("text2text-generation", model="google/flan-t5-small")

@app.get("/infer_t5")
def t5(input):
    output = pipe_flan(input)
    return {"output": output[0]["generated_text"]}
```

2. 编写`index.html`，得到一个包含页面代码的简单表单。

```html
<main>
  <section id="text-gen">
    <h2>Text generation using Flan T5</h2>
    <p>
      Model:
      <a
        href="https://huggingface.co/google/flan-t5-small"
        rel="noreferrer"
        target="_blank"
        >google/flan-t5-small
      </a>
    </p>
    <form class="text-gen-form">
      <label for="text-gen-input">Text prompt</label>
      <input
        id="text-gen-input"
        type="text"
        value="German: There are many ducks"
      />
      <button id="text-gen-submit">Submit</button>
      <p class="text-gen-output"></p>
    </form>
  </section>
</main>
```3. 在`main.py`文件中挂载静态文件并在根路由中显示html文件

```python
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def index() -> FileResponse:
    return FileResponse(path="/app/static/index.html", media_type="text/html")
```

4. 在`script.js`文件中，让它处理请求

```javascript
const textGenForm = document.querySelector(".text-gen-form");

const translateText = async (text) => {
  const inferResponse = await fetch(`infer_t5?input=${text}`);
  const inferJson = await inferResponse.json();

  return inferJson.output;
};

textGenForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const textGenInput = document.getElementById("text-gen-input");
  const textGenParagraph = document.querySelector(".text-gen-output");

  textGenParagraph.textContent = await translateText(textGenInput.value);
});
```

5. 授予正确目录权限

正如[Permissions Section](./spaces-sdks-docker#permissions)中所讨论的，容器以用户ID 1000运行。这意味着该空间可能面临权限问题。例如，`transformers`下载并缓存`HF_HOME`路径下的模型。解决此问题的最简单方法是创建具有正确权限的用户并使用它来运行容器应用程序。我们可以通过将以下行添加到 `Dockerfile` 来做到这一点。

```Dockerfile
# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
```

最终的`Dockerfile`应该是这样的：

```Dockerfile

# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

成功了！您的应用程序现在应该可以运行了！查看[DockerTemplates/fastapi_t5](https://huggingface.co/spaces/DockerTemplates/fastapi_t5)查看最终结果。

多么美妙的旅程啊！请记住，Docker Spaces 为您提供了很大的自由，因此您不限于使用 FastAPI。从[Go Endpoint](https://huggingface.co/spaces/DockerTemplates/test-docker-go)到[Shiny App](https://huggingface.co/spaces/DockerTemplates/shiny-with-python)，极限就是月亮！看看[some official examples](./spaces-sdks-docker-examples)。如果需要，您还可以将 Space 升级为 GPU 😃

## 调试

您可以通过检查 **Build** 和 **Container** 日志来调试您的 Space。单击 **打开日志** 按钮打开模式。

如果一切顺利，您将在 **Build** 选项卡上看到 `Pushing Image` 和 `Scheduling Space`在**容器**选项卡上，您将看到应用程序状态，在本例中为`Uvicorn running on http://0.0.0.0:7860`

此外，您可以在您的空间上启用开发模式。开发模式允许您通过 VSCode 或 SSH 连接到正在运行的空间。在这里了解更多信息：https://huggingface.co/dev-mode-explorers

## 阅读更多

- [Docker Spaces](spaces-sdks-docker)
- [List of Docker Spaces examples](spaces-sdks-docker-examples)

### 空间作为 MCP 服务器
https://huggingface.co/docs/hub/spaces-mcp-servers.md