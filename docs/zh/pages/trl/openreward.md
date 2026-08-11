<!-- huggingface-docs: machine-translated zh-CN from English source -->

# OpenReward 集成，用于培训法学硕士与环境

[OpenReward](https://openreward.ai) 是一个基于 [Open Reward Standard (ORS)](https://openrewardstandard.io) 的 RL 环境的开放生态系统，[Open Reward Standard (ORS)](https://openrewardstandard.io) 是一种与语言无关的公共 HTTP/SSE 协议，用于确定环境如何公开其任务、工具、会话和奖励。因为 ORS 只是一个协议，所以相同的环境可以在 [OpenReward platform](https://openreward.ai) 上运行，在任何容器服务上自托管，或者在 `localhost` 上本地进行开发。 [openreward.ai](https://openreward.ai) 提供了即用型环境目录。

本指南涵盖**如何将 OpenReward 与 TRL 集成**。有关标准本身的更多信息，请参阅[ORS docs](https://docs.openreward.ai/)。

> [!注意]
> 集成位于`trl.experimental.openreward`，并在`trl[openreward]`额外的后面（延迟导入 - 非用户无需支付任何费用）。

## 何时使用 OpenReward 环境

[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 通过 `environment_factory` 插槽支持基于环境的培训 — 参见 [OpenEnv](openenv) 了解总合同。当您想要针对 ORS 环境进行训练时，请使用 OpenReward：[OpenReward catalog](https://openreward.ai)（例如 `Eigent/SETA`、`kanishk/EndlessTerminals`、`nebius/SWE-rebench-V2`）、您在自己的基础设施上自行托管的环境或您正在开发的本地服务器。

## 安装

```bash
pip install trl[openreward]
```

这将安装 `openreward` Python SDK。集成本身会延迟导入`openreward`，因此不接触`trl.experimental.openreward`的用户不会受到影响。

## 快速开始`OpenRewardSpec` 类通过将 1:1 映射到这些 kwarg 名称的属性公开，将单个 ORS 环境连接到三个 TRL 训练器槽位 — `train_dataset`、`environment_factory`、`reward_funcs`：

```python
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openreward import OpenRewardSpec

spec = OpenRewardSpec("Eigent/SETA", num_tasks=64)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=GRPOConfig(
        num_generations=2,
        max_steps=5,
        max_tool_calling_iterations=20,
        log_completions=True,
    ),
    train_dataset=spec.train_dataset,
    environment_factory=spec.environment_factory,
    reward_funcs=spec.reward_funcs,
)
trainer.train()
```

在底层，`OpenRewardSpec` 会在第一次访问时惰性地执行三件事：

1. **`spec.train_dataset`**：从环境的任务列表中派生出`datasets.Dataset`（通过 SDK 进行一次 HTTP 往返）。至少有 `prompt`、`task_index`，以及折叠的每个任务元数据列。
2. **`spec.environment_factory`**：返回一个零参数可调用对象，该可调用对象在每次调用时都会生成一个新的每转出适配器。该适配器为每个 ORS 工具公开一个 Python 方法，并具有从环境的 JSON 架构自动生成的类型化签名和文档字符串。 TRL 的工具收集者通过 `inspect.getmembers` 拾取它们。
3. **`spec.reward_funcs`**：仅结果奖励函数（轨迹中最后一个非空奖励），适合像 SETA 这样的稀疏奖励环境。

## 使用 hub 环境

传递 [openreward.ai](https://openreward.ai) 目录名称作为目标。 SDK从环境中读取`OPENREWARD_API_KEY`进行身份验证。

```python
spec = OpenRewardSpec("Eigent/SETA", num_tasks=64)
```

## 使用自托管环境

直接传递网址。如果您的服务器不强制执行 API 密钥，则不需要 API 密钥。

```python
spec = OpenRewardSpec("https://my-org-my-env.hf.space", env_name="my_env")
```> [!重要]
> 默认情况下，`openreward` SDK 需要两个子域平台布局（`api.<host>` 用于无状态调用，`sessions.<host>` 用于基于 SSE 的会话调用）。对于**单主机**自托管服务器（一个 URL 服务所有内容），请在构建 `OpenRewardSpec` 之前设置下面的覆盖环境变量：
>
> ```python
> import os
>
> URL = "https://my-org-my-env.hf.space"
> os.environ["OPENREWARD_API_URL"]     = URL
> os.environ["OPENREWARD_SESSION_URL"] = URL
>
> spec = OpenRewardSpec(URL, env_name="my_env")
> ```

## 在本地运行最小环境

在没有外部依赖的情况下尝试端到端集成的最快方法是使用 `openreward` SDK 的 `Environment` + `Server` 脚手架定义的小型 ORS 服务器。下面的示例是一个完整的 `echo` 环境 - 模型通过使用任务的目标字符串调用 `echo(text=...)` 来获胜。

```python
# server.py
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

class EchoTaskSpec(BaseModel):
    target: str

class EchoParams(BaseModel):
    text: str

class EchoEnvironment(Environment):
    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = EchoTaskSpec.model_validate(task_spec)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        return [{"target": "hello"}, {"target": "world"}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(type="text", text=f"Echo '{self.config.target}' to win.")]

    @tool
    async def echo(self, params: EchoParams) -> ToolOutput:
        """Submit a string. Reward 1.0 + finished if it matches the target.

        Args:
            text: The string to echo back.
        """
        correct = params.text == self.config.target
        return ToolOutput(
            blocks=[TextBlock(type="text", text="match" if correct else "no match")],
            reward=1.0 if correct else 0.0,
            finished=correct,
        )

if __name__ == "__main__":
    Server([EchoEnvironment]).run(host="0.0.0.0", port=8000)
```

运行它：

```bash
pip install openreward fastapi uvicorn pydantic
python server.py     # listens on :8000
```

然后将 `OpenRewardSpec` 指向它（使用上面描述的 URL 覆盖）：

```python
import os
URL = "http://127.0.0.1:8000"
os.environ["OPENREWARD_API_URL"]     = URL
os.environ["OPENREWARD_SESSION_URL"] = URL

from trl.experimental.openreward import OpenRewardSpec
spec = OpenRewardSpec(URL, env_name="echoenvironment")
print(spec.train_dataset)        # 2 rows, task_index + target columns
```

这也是 TRL 自己的测试所使用的固定模式 - 有关已部署的 Space，请参阅[⟦T43⟧](https://huggingface.co/spaces/trl-internal-testing/openreward-echo-env)。

## 选择任务

`OpenRewardSpec` 接受计数或显式索引列表：

```python
spec = OpenRewardSpec("Eigent/SETA", num_tasks=10)                      # first 10 tasks
spec = OpenRewardSpec("Eigent/SETA", indices=[0, 5, 13, 27])            # specific indices
spec = OpenRewardSpec("Eigent/SETA", indices=list(range(50, 100)))      # range
```

`num_tasks` 和 `indices` 是互斥的，两者都只获取它们需要的任务（没有完整的任务列表扫描）。

## 工具绑定如何工作在构建时，规范调用环境的 `/tools` 端点来获取工具规范列表（每个规范都有名称、描述和用于参数的 JSON 架构）。对于每个工具，它都会在每次部署适配器上生成一个 Python 方法，该方法具有类型化签名和从架构派生的文档字符串。因此，`transformers.utils.get_json_schema` 和 TRL 的 `inspect.getmembers(env, ismethod)` 都可以为模型生成正确的工具模式，而无需每个环境的包装器代码。

如果工具描述包含无法安全拼接到 Python 源代码中的字符，则绑定器会回退到经过清理的形式，因此绑定在真实环境上永远不会失败。

## 奖励函数

`spec.reward_funcs` 默认为仅结果奖励 - 对于每次推出，它都会返回在轨迹期间观察到的最后一个非空奖励。这是稀疏奖励环境的正确默认值（例如 SETA，其中只有 `submit_solution` 返回非空奖励）。

如果想要自定义奖励，就写一个常规的TRL奖励函数，直接传递：

```python
def my_reward(environments, **kwargs) -> list[float]:
    return [env.reward * 2.0 for env in environments]   # double the env reward, etc.

trainer = GRPOTrainer(
    ...,
    reward_funcs=my_reward,
)
```

每次推出适配器公开 TRL 所需的运行状态 - `env.reward`、`env.rewards`、`env.metadata`、`env.finished`、`env.last_output` - 用于任意事后奖励塑造。

## OpenRewardSpec[[trl.experimental.openreward.OpenRewardSpec]]- **目标** (*str*) --
  openreward.ai 目录名称（*“Eigent/SETA”*）或指向任何 ORS 服务器的 URL
  （*“https://you-seta.hf.space”*，*“http://localhost:8080”*）。通过 *://* 中的存在自动检测
  字符串。
- **num_tasks** (*int*, *可选*) --
  限制拉入数据集中的任务数量。 `None` 使用 env 公开的每个任务。
- **split**（*str*，*可选*，默认为*“train”*）--
  从哪个 split 的任务列表中提取。
- **索引** (*list[int]*, *可选*) --
  要训练的具体任务指标。与`num_tasks`互斥。对于调试或
  课程子集。
- **api_key** (*str*, *可选*) --
  `OPENREWARD_API_KEY` 覆盖。仅当 `target` 是目录名称时使用。
- **秘密** (*dict[str, str]*, *可选*) --
  每个会话的秘密转发到`env.session(secrets=)`。
- **env_name** (*str*, *可选*) --
  覆盖要在服务器上查找的环境名称。很少需要。
- **include_metadata**（*bool*，*可选*，默认为*True*）--
  将每个任务的元数据（*难度*、*类别*、*标签*、...）折叠到数据集行中，以便奖励函数可以
  通过 TRL 的 `inputs` 参数来阅读它们。
- **discover_task_tools**（*bool*，*可选*，默认为*True*）--如果 `True`，则打开一个短暂的 ORS 会话并使用 `session.list_tools()` 以便特定于任务的工具
  （`GET …/task_tools` 按 ORS — 例如 `@tool(shared=False)` 和 `list_task_tools()`）绑定到 GRPO。
  如果探测失败，则仅回退到`environment.list_tools()`。设置`False`跳过额外的会话
  （仅限共享工具/离线怪癖）。
- **task_tools_discovery_index** (*int*, *可选*) --
  设置时任务索引仅用于发现会话； **覆盖**下面的多索引探测。当
  省略并设置 `indices=`，发现会为 `indices` 中的每个 **不同** 条目打开一个探测会话
  （已排序）并按名称**合并**工具规范，以便绑定每个列出的任务中的特定于任务的工具。当
  省略并使用`num_tasks`/完整列表模式，仅探测任务`0`。被忽略时
  `discover_task_tools=False`。
将 ORS 环境连接到 TRL 训练器的单个规范对象。

## 限制

- 集成在 `trl.experimental` 中 — API 可能会发生变化。设置 `TRL_EXPERIMENTAL_SILENCE=1` 以消除 CI 日志中的警告。
- 目前公开了覆盖一个环境的单个`OpenRewardSpec`；尚不支持多环境训练（类似于 OpenEnv“元环境”模式）。
- 长时间运行的部署（每集超过 15 分钟）需要保持连接 ping - 尚未连接。

## 参考

- [Open Reward Standard](https://openrewardstandard.io)
- [OpenReward platform](https://openreward.ai)
- [⟦T78⟧ Python SDK](https://pypi.org/project/openreward/)
- [Echo env Space — ⟦T79⟧](https://huggingface.co/spaces/trl-internal-testing/openreward-echo-env)### 工作培训
https://huggingface.co/docs/trl/v1.9.2/jobs_training.md