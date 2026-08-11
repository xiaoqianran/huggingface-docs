<!-- huggingface-docs: machine-translated zh-CN from English source -->

# OpenEnv 集成，用于培训法学硕士与环境

[OpenEnv](https://github.com/huggingface/OpenEnv) 是一个开源框架，用于定义、部署强化学习 (RL) 和代理工作流程中的环境并与之交互。它为环境交互提供标准化的API，并支持作为后端服务器的运行环境（通过WebSocket或容器化执行）。您可以在 [Hugging Face Hub](https://huggingface.co/collections/openenv/openenv-environment-hub) 上找到一系列即用型 OpenEnv 环境。

本指南涵盖**如何将 OpenEnv 与 TRL 集成**。有关 OpenEnv 本身的更多信息，请参阅 [OpenEnv docs](https://huggingface.co/docs/openenv)。

> [!注意]
> 您可以在示例概述中探索即用示例 [scripts](example_overview#openenv-scripts) 和 [notebooks](example_overview#openenv-notebooks)。

## 何时使用环境

[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)可用于培训代理。对于代理任务，它支持两种模式：**工具**，其中模型可以调用外部函数，但每次调用都是无状态且独立的；以及**环境**，它在轮次之间维护状态，从而实现真正的多轮交互，其中代理的行为塑造未来的观察结果。当连续性很重要时使用环境 - 例如，导航游戏、浏览网页或代理接下来看到的内容取决于它之前所做的任何任务。## 选择环境集成

OpenEnv 是此处记录的本机路径。两个进一步的集成 - [OpenReward](openreward) 和 [Harbor](harbor) - 符合相同的 `environment_factory` 合同，并且可以在 TRL 级别互换。请参阅 GRPO 指南中的[comparison of environment integrations](grpo_trainer#agent-training)，选择其生态系统适合您的任务的生态系统。

## 安装

OpenEnv 环境托管为 Hugging Face Spaces，它们也是可通过 pip 安装的 Git 存储库：

```bash
# Echo environment
pip install "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"

# Wordle (TextArena) environment
pip install "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle"

# Catch (OpenSpiel) environment
pip install "openenv-openspiel-env @ git+https://huggingface.co/spaces/openenv/openspiel_env"
```

这将安装通过 WebSocket 与远程环境服务器通信的 **环境客户端**（例如 `EchoEnv`），以及操作/观察模型和所有必需的依赖项（包括 `openenv`）。

> [!提示]
> 您可以在 HF Space 页面上找到适用于任何环境的安装命令。单击 **⋮（三个点）** 菜单并选择 **“使用此空间”** 以查看安装说明。

> [!提示]
> 您还可以使用 `pip install "openenv[core]>=0.3.1"` 从 PyPI 安装核心包，但请注意，可能需要单独安装特定于环境的依赖项。

对于开发，您可以克隆 OpenEnv 存储库并在本地安装：

```bash
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv/envs/echo_env
pip install -e .
```> [!注意]
> TRL 中的每个环境脚本都包含内联依赖元数据 (PEP 723)，因此您也可以使用 [uv](https://docs.astral.sh/uv/) 直接运行它们：
>
> ```bash
> uv run examples/scripts/openenv/echo.py
> ```
>
> 这会自动在隔离的虚拟环境中安装所需的环境包。

## 快速开始

了解集成的最快方法是一个完整的示例。 [echo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py) 脚本使用 [Echo environment](https://huggingface.co/docs/openenv/environments/echo) 训练模型，该模型根据文本长度奖励完成情况：

```python
from datasets import Dataset
from echo_env import EchoEnv
from echo_env.models import EchoAction

from trl import GRPOConfig, GRPOTrainer

ENV_URL = "https://openenv-echo-env.hf.space"

class EchoToolEnv:
    def __init__(self):
        self.env = EchoEnv(base_url=ENV_URL)
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        return None

    def echo(self, message: str) -> str:
        """
        Echo the message back from the environment.

        Args:
            message: The message to echo

        Returns:
            The echoed message.
        """
        observation = self.env.step(EchoAction(message=message))
        self.reward = observation.observation.reward
        return observation.observation.echoed_message

def reward_func(environments, **kwargs):
    return [env.reward for env in environments]

dataset = Dataset.from_dict(
    {"prompt": [[{"role": "user", "content": "Try to echo 'Hello World!' in the environment."}]] * 64}
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=GRPOConfig(
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    ),
    environment_factory=EchoToolEnv,
)
trainer.train()
```

就是这样。以下是幕后发生的事情：1. **`environment_factory=EchoToolEnv`**：训练器每代创建一个`EchoToolEnv`实例（传递类，而不是实例）。
2. **`reset()`** 在每集开始时调用以初始化状态。返回观察字符串（或`None`）。
3. **工具发现**：训练器发现环境实例（此处为`echo()`）上的所有公共方法，并将它们公开为函数调用工具。每个方法都必须有一个带有类型参数的正确文档字符串，培训师用它来构建工具模式。
4. **多轮循环**：训练器生成补全，解析工具调用，执行`echo()`，追加结果，再次生成，直到模型停止调用工具或达到`max_completion_length`。
5. **奖励功能**：在剧集结束后（环境重置之前）从每个环境实例中读取`env.reward`。

```bash
# Run the example
python examples/scripts/openenv/echo.py

# Customize model and environment URL
python examples/scripts/openenv/echo.py --model Qwen/Qwen3-0.6B --env-host https://openenv-echo-env.hf.space
```

以下是训练的奖励曲线：

> [!注意]
> 您可以在示例概述中探索更多即用型示例 [scripts](example_overview#openenv-scripts) 和 [notebooks](example_overview#openenv-notebooks)。

## `environment_factory` 的工作原理TRL 的 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 通过 `environment_factory` 参数支持交互式环境训练。当提供时，训练器会自动处理多轮工具调用循环：它生成补全，解析工具调用，针对环境执行它们，并将结果反馈给模型。全部无需自定义推出代码。

### 环境等级要求

您的环境类别必须遵循以下规则：

- `__init__(self)` *（可选）*：如果提供，则不得带任何参数。用它来初始化状态或客户端。如果您需要外部配置（例如 URL），请从封闭范围或模块级变量中捕获它。
- `reset(self, **kwargs)`：在每集开始时调用。接收所有数据集列作为关键字参数。返回一个字符串观察值（如果没有初始观察值，则返回 `None`）。
- **工具方法**：除`reset`之外的任何公共方法（不以`_`开头）都会自动公开为工具。每个工具方法都必须有一个带有 `Args:` 描述的文档字符串，因为培训师使用这些来生成模型的工具架构。

### 环境类提示- **奖励状态**：您可以在环境实例上存储您想要的任何状态（例如，`self.reward`、`self.done`等），并通过`environments`参数在奖励函数中访问它。有关此模式的示例，请参阅[Quick Start guide](#quick-start)。
- **错误处理**：如果工具方法引发异常（例如，`ValueError("Game over.")`），训练器会捕获该异常并将错误消息作为工具响应反馈给模型。这是指示操作无效或剧集已结束的推荐方式。

```python
ENV_URL = "https://my-env.hf.space"

class MyEnv:
    def __init__(self):
        self.client = MyClient(base_url=ENV_URL)  # captured from enclosing scope
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        return "Initial observation for the model"

    def my_tool(self, arg1: str, arg2: int) -> str:
        """
        Description of what this tool does.

        Args:
            arg1: Description of arg1
            arg2: Description of arg2

        Returns:
            The result message.
        """
        self.reward = 1.0
        return "Tool result"
```

> [!重要]
> 工具必须是**单独的方法**，具有描述性名称和类型参数（例如，`guess(word: str)`、`move(direction: str)`）。我们不建议使用像`step(action)`这样的通用方法，因为模型需要有意义的工具名称和参数描述来学习工具调用。

### 奖励功能

奖励函数接收 `environments` 参数（环境实例列表），因此您可以访问该情节期间存储的任何状态：

```python
def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
```

有关奖励函数的更多信息，请参阅[GRPO - Custom Reward Functions](grpo_trainer#using-a-custom-reward-function)。

### 奖励函数的提示

我们发现在使用 OpenEnv 环境和 GRPO 时有一些有用的东西：- **简单的奖励效果很好。** 在我们对 Wordle 和数独的实验中，二元奖励（1.0 表示成功，0.0 表示成功）比具有部分学分的形状奖励提供了更清晰的训练信号。 GRPO 比较组内的完成情况，因此相对排名比绝对值更重要。
- **检查最终状态，而不是路径。** 如果可能，让环境判断结果（例如，“模型是否解决了难题？”），而不是检查它是否遵循特定的操作序列。这使得模型可以自由地发现自己的策略。
- **在训练前测试您的奖励。** 手动运行几集（请参阅[Wordle example notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb)）以确认环境返回合理的奖励。如果有能力的模型得分不能高于随机基线，则奖励信号可能需要调整。

### `max_completion_length` 多回合剧集

`max_completion_length`参数限制**整个多轮对话**的令牌总数（所有模型生成+工具结果的组合），而不仅仅是单个生成。对于多回合的环境（例如，有数十步的数独），您可能需要增加它：

```python
args = GRPOConfig(
    max_completion_length=4096,  # default is usually 256-1024, increase for long episodes
    # ...
)
```如果剧集被缩短（模型在游戏中停止），这可能是原因。

## 高级示例：Wordle

让我们训练一个模型来使用 [⟦T55⟧](https://huggingface.co/docs/openenv/environments/textarena) 环境来玩 [Wordle](https://www.nytimes.com/games/wordle/index.html)。这演示了多轮交互、累积反馈处理以及通过异常终止事件。

> [!注意]
> 您可以在[the OpenEnv Wordle GRPO example](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb)中探索此示例的笔记本版本。

### TextArena 环境

[TextArena](https://huggingface.co/papers/2504.11442) 是一个基于文本的竞争性游戏的开源集合，旨在使用 Wordle、Snake、Tic-Tac-Toe 等文本游戏来评估法学硕士的推理技能。

![image of TextArena](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/text_arena_evals.png)

### 为什么是Wordle？

Wordle 是基于环境的 RL 的一个很好的基准，因为它需要对反馈进行推理，是纯粹基于文本的，并且来自 1B 参数的模型可以对此进行改进。每个猜测只有 8 个标记，因此实验起来很轻量。

> [!NOTE] Wordle 是如何工作的？
> Wordle 是一款猜词游戏，玩家必须在 6 次尝试中猜出 5 个字母的单词。每次猜测后，环境都会提供逐字反馈：
>
> ```
> G U E S S
> X G Y X X
> ```
> X = 不在单词中，G = 正确位置（绿色），Y = 错误位置（黄色）。这里，“U”是正确的并且在适当的位置，“E”在单词中但是放错了位置。### 环境等级

`WordleEnv` 类包装 TextArena 客户端并将 `guess()` 公开为工具：

```python
from textarena_env import TextArenaAction, TextArenaEnv

class WordleEnv:
    def __init__(self):
        self.client = TextArenaEnv(base_url="https://openenv-wordle.hf.space")

    def reset(self, **kwargs) -> str | None:
        result = self.client.reset()
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        result = self.client.step(TextArenaAction(message=guess))
        _full_feedback = result.observation.messages[0].content
        feedback = _full_feedback[len(self._last_full_feedback):]
        self._last_full_feedback = _full_feedback
        if "You attempted an invalid move" in feedback:
            self.reward = 0.0
        else:
            self.reward = result.reward
        self.done = result.done
        return feedback
```

关键设计选择：

- **`reset()`** 返回初始游戏消息作为模型看到的第一个观察结果。
- **`guess()`** 是唯一的工具。该模型每回合都会用 5 个字母的单词来称呼它。
- **累积反馈切片**：TextArena 每回合返回完整的游戏历史记录。我们只切掉新的部分以避免重复上下文。
- **完成时的异常**：如果模型在游戏结束后尝试猜测，`guess()`会引发`ValueError`。培训师捕捉到这一点并将 `"Game over."` 作为工具响应反馈给模型。该模型学会在收到此信号后停止调用工具。

### 奖励功能和训练

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]

prompt = """You are an expert Wordle solver with deep knowledge of English vocabulary...
Use the tool `guess` to make a guess."""

dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * 1000})

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        chat_template_kwargs={"enable_thinking": False},
        max_completion_length=1024,
        num_generations=4,
        gradient_accumulation_steps=64,
    ),
    environment_factory=WordleEnv,
)
trainer.train()
```

如果模型获胜，则环境返回 `1.0`，否则返回 `0.0`。

### 运行示例

**并置模式（1 个 GPU，推荐）**

```bash
python examples/scripts/openenv/wordle.py --vllm-mode colocate
```

这在与训练相同的过程中运行 vLLM，仅需要一个 GPU。

**服务器模式（2 个以上 GPU，可扩展）**

```bash
# Terminal 1: Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000

# Terminal 2: Run GRPO training with OpenEnv
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py --vllm-mode server --vllm-server-url http://localhost:8000
```

### 结果

该模型通过减少重复和增加正确猜测来提高其性能。然而，使用`enable_thinking=False`的Qwen3-1.7B并不能持续赢得比赛。> [!注意]
> 使用`enable_thinking=False`（这些示例中的默认值），像 Qwen3-1.7B 这样的小模型可以学习改进他们的猜测，但不应期望能够始终如一地解决游戏。为了获得更好的结果，请使用更大的模型或启用思维模式 (`enable_thinking=True`)，该模式允许模型在猜测较长完成时间的成本之前进行推理。

我们尝试了像[⟦T68⟧](https://huggingface.co/openai/gpt-oss-20b)这样的更大模型，发现它能够持续赢得比赛，尽管这需要更多的计算。

## 多环境训练

您可以同时跨多个环境训练单个模型。当您希望模型并行学习不同的技能时，这非常有用。例如，在同一次训练中玩 Wordle（语言推理）和 Catch（空间推理）。

关键思想是创建一个**元环境类**，它包装多个环境并使用数据集列将每个样本路由到正确的环境。

### 它是如何工作的1. 将 `"env"` 列（或类似列）添加到数据集中，用于标识每个样本所属的环境。
2. 在`reset(**kwargs)`中，阅读`kwargs["env"]`来选择该剧集的活动环境。
3.暴露所有环境中的工具；培训师发现所有公共方法。
4. 每个环境使用单独的奖励函数，对于不属于该环境的样本返回`None`。 TRL 使用 `nansum`/`nanmean` 处理 `None` 值。

### 示例：Wordle + Catch

[multi_env.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/multi_env.py) 脚本同时在 Wordle 和 Catch 上进行训练：

```python
class MultiEnv:
    def __init__(self):
        self._wordle_client = None
        self._catch_client = None
        self.active = None
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        self.active = kwargs.get("env", "wordle")
        self.reward = 0.0
        self.done = False

        if self.active == "wordle":
            if self._wordle_client is not None:
                try:
                    self._wordle_client.close()
                except Exception:
                    pass
            self._wordle_client = TextArenaEnv(base_url=WORDLE_URL)
            result = self._wordle_client.reset()
            self._last_full_feedback = result.observation.messages[0].content
            self.reward = 0.0
            return self._last_full_feedback
        elif self.active == "catch":
            if self._catch_client is not None:
                try:
                    self._catch_client.close()
                except Exception:
                    pass
            self._catch_client = OpenSpielEnv(base_url=CATCH_URL)
            result = self._catch_client.reset()
            self.done = result.observation.done
            return _format_catch_obs(result.observation.info_state)

    # Wordle tool
    def guess(self, guess: str) -> str:
        """Make a guess in the Wordle environment. ..."""
        ...

    # Catch tools
    def move(self, direction: str) -> str:
        """Move the paddle left or right. ..."""
        ...

    def stay(self) -> str:
        """Do nothing and let the ball fall one step. ..."""
        ...
```

关键模式：- **延迟客户端初始化**：在`reset()`而不是`__init__()`中创建客户端，以避免不必要的 WebSocket 连接。
- **重新打开前关闭**：在创建新客户端之前关闭之前的客户端，以避免服务器容量错误。
- **`kwargs` 路由**：数据集中的 `"env"` 列作为关键字参数传递到 `reset()`。
- **所有工具同时公开**：无论活动环境如何，模型都会将 `guess`、`move` 和 `stay` 视为可用工具。如果它调用了错误的工具（例如，Wordle 期间的`move`），该方法会引发一个`ValueError`，训练师可以优雅地捕捉到。在实践中，模型学习根据系统提示使用正确的工具。

### 每个环境的奖励函数

对于来自其他环境的样本，每个奖励函数都会返回 `None`：

```python
def wordle_reward(environments, **kwargs) -> list[float | None]:
    return [env.reward if env.active == "wordle" else None for env in environments]

def catch_reward(environments, **kwargs) -> list[float | None]:
    rewards = []
    for env in environments:
        if env.active != "catch":
            rewards.append(None)
        elif env.done:
            rewards.append(max(env.reward, 0.0))
        else:
            rewards.append(0.0)
    return rewards
```

TRL在内部将`None`转换为`nan`，并使用`nansum`/`nanmean`进行聚合，因此每个样本仅根据其相关的奖励函数进行评分。

### 具有环境路由的数据集

```python
n = 500
dataset = Dataset.from_dict({
    "prompt": (
        [[{"role": "user", "content": wordle_prompt}]] * n
        + [[{"role": "user", "content": catch_prompt}]] * n
    ),
    "env": ["wordle"] * n + ["catch"] * n,
})
```

### 运行多环境示例

```bash
python examples/scripts/openenv/multi_env.py \
    --wordle-url https://openenv-wordle.hf.space \
    --catch-url https://openenv-openspiel-env.hf.space \
    --vllm-mode colocate \
    --gradient-accumulation-steps 4 \
    --num-generations 8
```> [!提示]
> 跨多个环境进行训练时，监控每个奖励函数指标（`train/reward_func_0`、`train/reward_func_1` 等），而不是组合的 `train/reward`。组合的指标在环境之间交替，并且可能显得嘈杂。

## 运行环境

使用`environment_factory`时，训练器会自动连接到环境服务器。您只需要运行服务器即可。运行 OpenEnv 环境服务器有三种方法：

**连接到远程 Hugging Face Space** *（最简单）*

大多数示例脚本默认为托管空间（无需设置）：

```python
env = EchoEnv(base_url="https://openenv-echo-env.hf.space")
```

> [!警告]
> 对于训练，**将空间复制到您自己的帐户**以避免并发问题。训练器同时打开 N 个 WebSocket 连接（每代一个），而共享空间可能不支持这一点。详情请参阅[Server concurrency](#server-concurrency)。

**Docker 容器** *（推荐用于生产）*

```bash
docker run -d -p 8001:8000 --platform linux/amd64 registry.hf.space/openenv-echo-env:latest
```

然后连接：

```python
env = EchoEnv(base_url="http://0.0.0.0:8001")
```

我们将端口 8001 映射到 8000，以使端口 8000 可用于 vLLM 服务器。

您还可以以编程方式启动容器：

```python
env = EchoEnv.from_docker_image("registry.hf.space/openenv-echo-env:latest")
```

> [!注意]
> 您可以在 Hub 上找到任意 Space 的 Docker 镜像：打开 Space 页面 → **⋮（三个点）** → **“本地运行。”**
>
> ![open_env_launch_docker](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/open_env_launch_docker.png)**本地Python进程** *（用于开发）*

```bash
hf download openenv/echo_env --repo-type=space --local-dir=echo_env
python -m uvicorn echo_env.src.envs.echo_env.server.app:app --host 0.0.0.0 --port 8001
```

然后连接：

```python
env = EchoEnv(base_url="http://0.0.0.0:8001")
```

欲了解更多详情，请参阅[OpenEnv catalog](https://huggingface.co/docs/openenv/environments)。

## 环境目录

探索当前维护环境目录的最佳方法是访问官方 OpenEnv [catalog](https://huggingface.co/collections/openenv/environment-hub)。

要创建您自己的环境，请查看 [Building Your Own Environment with OpenEnv](https://huggingface.co/docs/openenv/getting_started/environment-builder) 上的指南。环境与 Hub 紧密集成，因此您可以推送新环境供社区重用。

## 服务器并发

当使用`environment_factory`时，训练器创建N个环境实例（每代一个），每个实例打开一个到服务器的WebSocket连接。默认情况下，OpenEnv 服务器仅允许 1 个并发会话，这将导致训练期间失败。

要支持并行训练，请配置服务器的并发性：

1. 在您的环境文件中，声明并发会话支持：
```python
SUPPORTS_CONCURRENT_SESSIONS: bool = True
```

2. 在您的服务器应用程序中，设置并发限制：
```python
app = create_app(
    create_my_environment,
    MyAction,
    MyObservation,
    max_concurrent_envs=64,  # match or exceed generation_batch_size
)
```

> [!提示]
> `max_concurrent_envs` 应 ≥ `generation_batch_size`（默认为 `per_device_train_batch_size × gradient_accumulation_steps`）。例如，对于 `gradient_accumulation_steps=64` 和批量大小 1，您至少需要 64 个并发会话。

## `environment_factory` 与 `rollout_func`

[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 支持两种基于环境的训练方法：- **`environment_factory`**（推荐）：您使用工具方法定义环境类，训练器自动处理生成、工具调用解析和多轮循环。这是本指南通篇使用的方法。
- **`rollout_func`**：您自己编写整个生成和环境交互循环。这可以完全控制如何产生完成、如何执行工具以及如何计算奖励。

当 `environment_factory` 不适合您的用例时，请使用 `rollout_func`。例如，**外部代理服务器**，其中外部服务器拥有生成循环并管理其自己的代理环境交互协议。

### 使用统计数据收集
https://huggingface.co/docs/trl/v1.9.2/usage_stats.md