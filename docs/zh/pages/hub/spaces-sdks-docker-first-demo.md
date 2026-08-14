<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Your First Docker Space: Text Generation with T5

In the following sections, you'll learn the basics of creating a Docker Space, configuring it, and deploying your code to it.我们将使用 Docker 创建一个 **文本生成** 空间，用于演示 [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) 模型，该模型可以使用 FastAPI 作为服务器，在给定一些输入文本的情况下生成文本。

You can find a completed version of this hosted [here](https://huggingface.co/spaces/DockerTemplates/fastapi_t5).

## 创建一个新的 Docker 空间

We'll start by [creating a brand new Space](https://huggingface.co/new-space) and choosing **Docker** as our SDK.

Hugging Face Spaces are Git repositories, meaning that you can work on your Space incrementally (and collaboratively) by pushing commits. Take a look at the [Getting Started with Repositories](./repositories-getting-started) guide to learn about how you can create and edit files before continuing. If you prefer to work with a UI, you can also do the work directly in the browser.

当 [creating a new Space](https://huggingface.co/new-space) 时选择 **Docker** 作为 SDK，将通过在 `README.md` 文件的 YAML 块中将 `sdk` 属性设置为 `docker` 来初始化 Docker 空间。

```yaml
sdk: docker
```

You have the option to change the default application port of your Space by setting the `app_port` property in your `README.md` file's YAML block.默认端口为`7860`。

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

### 本地测试**Tip for power users (you can skip):** If you're developing locally, this is a good moment in which you can do `docker build` and `docker run` to debug locally, but it's even easier to push the changes to the Hub and see how it looks like!

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

如前所述，我们的想法是使用 Flan T5 模型进行文本生成。 We'll want to add some HTML and CSS for an input field, so let's create a directory called static with `index.html`, `style.css`, and `script.js` files.此时，你的文件结构应该如下所示：

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
```3. In the `main.py` file, mount the static files and show the html file in the root route

```python
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def index() -> FileResponse:
    return FileResponse(path="/app/static/index.html", media_type="text/html")
```

4. In the `script.js` file, make it handle the request

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

As discussed in the [Permissions Section](./spaces-sdks-docker#permissions), the container runs with user ID 1000. That means that the Space might face permission issues. For example, `transformers` downloads and caches the models in the path under the `HF_HOME` path. The easiest way to solve this is to create a user with righ permissions and use it to run the container application.我们可以通过将以下行添加到 `Dockerfile` 来做到这一点。

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

多么美妙的旅程啊！ Please remember that Docker Spaces give you lots of freedom, so you're not limited to use FastAPI.从[Go Endpoint](https://huggingface.co/spaces/DockerTemplates/test-docker-go)到[Shiny App](https://huggingface.co/spaces/DockerTemplates/shiny-with-python)，极限就是月亮！看看[some official examples](./spaces-sdks-docker-examples)。 You can also upgrade your Space to a GPU if needed 😃

## 调试

You can debug your Space by checking the **Build** and **Container** logs. Click on the **Open Logs** button to open the modal.

If everything went well, you will see `Pushing Image` and `Scheduling Space` on the **Build** tab在**容器**选项卡上，您将看到应用程序状态，在本例中为`Uvicorn running on http://0.0.0.0:7860`

此外，您可以在您的空间上启用开发模式。开发模式允许您通过 VSCode 或 SSH 连接到正在运行的空间。在这里了解更多信息：https://huggingface.co/dev-mode-explorers

## 阅读更多

- [Docker Spaces](spaces-sdks-docker)
- [List of Docker Spaces examples](spaces-sdks-docker-examples)

### 空间作为 MCP 服务器
https://huggingface.co/docs/hub/spaces-mcp-servers.md