<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 用于培训法学硕士与环境的 Harbor 集成

[Harbor](https://www.harborframework.com)是一个在沙箱中运行代理任务的框架。它将**任务**（指令+沙箱图像+验证器），**线束/代理**（工具表面+循环）和**沙箱**（`docker`，`e2b`，`daytona`，`gke`，...）解耦，因此它们可以自由混合。这使得它非常适合强化学习：相同的任务套件可以在您喜欢的沙箱后端上使用不同的工具表面进行训练。

本指南涵盖**如何将 Harbor 与 TRL 集成**。对于港口本身，请参阅[Harbor docs](https://www.harborframework.com/docs)。

> [!注意]
> 集成位于`trl.experimental.harbor`，并在`trl[harbor]`额外的后面（延迟导入 - 非用户无需支付任何费用）。

## 何时使用 Harbor 环境

[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 通过 `environment_factory` 插槽支持基于环境的培训 — 有关总合同，请参阅[OpenEnv](openenv)。当您想要针对 **Harbor 任务套件** 进行训练时，请使用 Harbor：任务的目录树，每个任务都有一个独立的沙箱 + 验证器（例如，一个数据分析代理套件，其中模型探索沙箱中的文件并编写评分者检查的答案）。

## 安装

```bash
pip install trl[harbor]
```> [!重要]
> Harbor通过vLLM驱动生成并使用`environment_factory`，这需要`vllm>=0.22.0`和`transformers>=5.2.0`。
>
> ```bash
> pip install 'vllm>=0.22.0'
> ```

这将安装 `harbor` 框架 (Python >= 3.12)。该集成会延迟导入 `harbor` 并在**进程中**运行它，因此不接触 `trl.experimental.harbor` 的用户不会受到影响。

还必须安装沙箱后端并在训练时可访问。 Harbor 将云后端保留在其自己的附加功能后面，因此请安装您想要使用的后端并提供其凭据：

```bash
pip install "harbor[e2b]"      # E2B cloud sandbox  -> environment_type="e2b",  needs E2B_API_KEY
# docker backend (environment_type="docker", Harbor's default) just needs a reachable Docker daemon
```

## 快速开始

`HarborSpec` 将单个 Harbor 任务套件连接到三个 TRL 训练器槽位 — `train_dataset`、`environment_factory`、`reward_funcs` — 通过公开将 1:1 映射到这些 kwarg 名称的属性：

```python
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.harbor import HarborSpec

spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", agent="bash", num_tasks=64)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=GRPOConfig(
        num_generations=8,
        max_steps=50,
        max_tool_calling_iterations=25,
        log_completions=True,
    ),
    train_dataset=spec.train_dataset,
    environment_factory=spec.environment_factory,
    reward_funcs=spec.reward_funcs,
)
trainer.train()
```

在底层，`HarborSpec` 会在第一次访问时惰性地执行三件事：1. **`spec.train_dataset`**：将任务套件解析到本地任务目录（如果需要，下载 HF 数据集）并使用 `prompt`（空 - 环境指令附加在 `reset`）、`task_dir`、`task_index` 以及每个任务构建`datasets.Dataset` `task.toml` 元数据列。
2. **`spec.environment_factory`**：返回一个零参数可调用，生成新的每次推出[HarborEnv](/docs/trl/v1.9.2/en/harbor#trl.experimental.harbor.HarborEnv)。在 `reset(task_dir)` 上，它启动任务的 Harbor 沙箱并返回其指令；工具方法执行到沙箱中； `env.reward` 在推出后运行验证程序一次。
3. **`spec.reward_funcs`**：每次推出时读取 Harbor 验证者标量的结果奖励。

## 数据集

`dataset` 可以是包含 Harbor 任务树的 Hugging Face 数据集存储库 ID，也可以是包含 `tasks/` 子树的本地路径。每个任务都是一个目录：

```
tasks/<task_id>/
├── instruction.md          # the task prompt (returned by reset)
├── task.toml               # config + metadata (gold answer, difficulty, ...)
├── environment/            # Dockerfile (+ any pre-agent data hooks)
└── tests/                  # test.sh / grader → writes the reward
```

选择具有`num_tasks`或`indices`（互斥）的子集：

```python
spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", num_tasks=10)        # first 10
spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", indices=[0, 5, 13])  # specific
```

## 代理：外部代理与安装代理

Harbor 支持代理驱动任务的两种方式，其区别决定了可以使用 RL 训练哪些内容：- [**External agents**](https://www.harborframework.com/docs/agents#external-agents) 在沙箱外部运行并驱动循环本身，通过 Harbor 的环境接口向容器发出命令（“通常通过 `exec` 方法执行 bash 命令”）。代理决定每个动作并解释每个结果；沙箱仅执行。
- [**Installed agents**](https://www.harborframework.com/docs/agents#installed-agents) 安装到*到容器镜像中*并作为无头子进程运行（扩展 `BaseInstalledAgent`）。 Harbor在沙箱内启动代理并解析其轨迹文件（`populate_context_post_run`）；代理根据自己的推理自主运行。**TRL 的集成是外部代理模式，目前仅支持该模式。** RL 训练需要训练器逐轮驱动推出：*正在训练的策略模型*生成每个回合，TRL 捕获其令牌和日志概率并应用环境掩码 - 正是`environment_factory` 通过黑盒`rollout_func` 提供的功能。安装的代理对此是不透明的：它使用自己的模型在容器内运行，并且仅在事后发出轨迹，因此没有策略令牌或日志概率供训练器优化，并且永远不会调用正在训练的模型。因此，[HarborEnv](/docs/trl/v1.9.2/en/harbor#trl.experimental.harbor.HarborEnv)是一个外部代理——它的工具方法`exec`进入沙箱，但循环和训练中的模型仍保留在TRL中。

## 选择基础代理（线束）

**基础代理**是工具 - 环境公开哪些工具方法以及它如何提交。使用 `agent=` 选择它：

```python
HarborSpec(dataset, agent="bash")                          # built-in single-bash-tool harness
HarborSpec(dataset, agent="my_pkg.harnesses:JupyterEnv")   # import path to your HarborEnv subclass
HarborSpec(dataset, agent="path/to/harness.py:JupyterEnv") # file path to your HarborEnv subclass
HarborSpec(dataset, agent=MyHarborEnv)                     # a HarborEnv subclass directly
```

内置的`"bash"`线束（[HarborBashEnv](/docs/trl/v1.9.2/en/harbor#trl.experimental.harbor.HarborBashEnv)）公开了一个`bash`工具，并通过编写`/workdir/answer.txt`来提交。两个更丰富的工具作为示例提供 - 每个都在其自己的文件夹中，并在 [⟦T54⟧](https://github.com/huggingface/trl/tree/main/examples/scripts/harbor/harnesses) 下有一个列出其工具的自述文件：- [⟦T55⟧](https://github.com/huggingface/trl/tree/main/examples/scripts/harbor/harnesses/jupyter) (`JupyterEnv`) — 有状态的 Python 内核（变量跨单元持久存在）+ shell 工具。
- [⟦T57⟧](https://github.com/huggingface/trl/tree/main/examples/scripts/harbor/harnesses/terminal_notes) (`TerminalNotesEnv`) — 6 个 shell 工具（包括后台进程）+ 4 个工具的持久注释工具包。

```python
HarborSpec(dataset, agent="examples/scripts/harbor/harnesses/jupyter/env.py:JupyterEnv")
HarborSpec(dataset, agent="examples/scripts/harbor/harnesses/terminal_notes/env.py:TerminalNotesEnv")
```

要编写自己的工具，请子类 [HarborEnv](/docs/trl/v1.9.2/en/harbor#trl.experimental.harbor.HarborEnv) 并添加工具方法 - 每个公共方法都成为一个工具（TRL 使用 `inspect.getmembers` 发现它们），因此为每个方法提供一个类型签名和一个文档字符串（用于构建工具架构）。保持助手以下划线为前缀。使用 `self._exec(cmd)` 在沙箱中运行 shell 命令，并设置 `PROMPT_SUFFIX` 将线束指导附加到任务指令中：

```python
from trl.experimental.harbor import HarborEnv

class GrepEnv(HarborEnv):
    PROMPT_SUFFIX = "\n\nUse `grep` and `read_file`. Submit by writing /workdir/answer.txt."

    def grep(self, pattern: str, path: str) -> str:
        """Search for `pattern` under `path`.

        Args:
            pattern: The regex to search for.
            path: The file or directory to search.
        """
        return self._exec(f"grep -rn {pattern!r} {path!r}")
```

## 沙箱后端

`environment_type` 直接传递到 Harbor（未经 TRL 验证）：

```python
HarborSpec(dataset, environment_type="e2b")   # cloud sandbox (offloads provisioning), needs E2B_API_KEY
HarborSpec(dataset, environment_type="docker")  # default; needs a local Docker daemon
```

建议使用 `e2b` 进行集群训练：只有 `environment.exec` 跨入云沙箱，因此 GPU 专用于该策略，并且您可以同时运行多个部署。

## 奖励函数

`spec.reward_funcs` 默认为结果奖励 - 每次部署它都会读取 Harbor 验证程序的标量 (`env.reward`)，该标量在部署后通过在沙箱中运行任务的 `tests/` 验证程序计算一次。对于自定义奖励，请编写常规 TRL 奖励函数：

```python
def my_reward(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
```

## API[[trl.experimental.harbor.HarborSpec]]- **数据集** (`str`) --
  包含 Harbor 任务树的 Hugging Face 数据集存储库 ID（例如
  `"AdithyaSK/data_agent_rl_environment_train"`)，或包含 `tasks/` 的目录的本地路径
  子树。每个任务都是一个带有 `instruction.md` / `task.toml` / `environment/` / `tests/` 的目录。
- **代理**（`str`或`type`，*可选*，默认为`"bash"`）--
  基础代理/线束——即环境暴露的工具表面。其中之一：内置名称 (`"bash"`)、
  导入路径 `"package.module:ClassName"`、文件路径 `"path/to/file.py:ClassName"` 或
  [HarborEnv](/docs/trl/v1.9.2/en/harbor#trl.experimental.harbor.HarborEnv) 直接子类化。
- **环境类型**（`str`，*可选*，默认为`"docker"`）--
  Harbor 沙箱后端，传递到 Harbor（无论它支持什么 - `"docker"`、`"e2b"`、`"daytona"`、
  `"gke"`、`"modal"`、`"runloop"`、...）。此处未验证；港口验证。 `"docker"` 是 Harbour 自己的
  默认；选择 `"e2b"` 将沙箱卸载到云端。
- **num_tasks** (`int`, *可选*) --
  限制拉入数据集中的任务数量。 `None` 使用树中的每个任务。
- **索引**（`list[int]`，*可选*）--
  具体任务索引（进入已排序的任务列表）。与`num_tasks`互斥。
- **include_metadata**（`bool`，*可选*，默认为`True`）--
  将每个任务 `task.toml` 元数据（gold_answer、难度等）折叠到数据集行中。将 Harbor 任务套件连接到 TRL 训练器的单个规范对象。

- **环境类型**（`str`，*可选*，默认为`"docker"`）--
  Harbor沙箱后端，传递到Harbor（`"docker"`，`"e2b"`，`"daytona"`，...）。
由 Harbor 沙箱 + 验证器支持的基本 TRL 环境。

子类定义工具方法（线束）。每次部署时生命周期 TRL 驱动：`reset(task_dir)`（开始
任务的沙箱，返回其指令） -> 工具方法（执行到沙箱中） -> `reward`（运行验证器
一次，懒惰地，在推出之后）。

单`bash`-工具吊带；通过写`/workdir/answer.txt`提交。

- **command** -- 要运行的 shell 命令。该命令是 stdout 和 stderr 的组合。

在沙箱中运行 shell 命令并返回其组合的 stdout+stderr。 shell 之间是无状态的
来电。用它来探索文件（ls，head，cat），运行Python（`python3 -c "..."`），并提交答案（`echo
-n "" > /workdir/answer.txt`)。

## 限制- 集成在 `trl.experimental` 中 — API 可能会发生变化。设置 `TRL_EXPERIMENTAL_SILENCE=1` 以消除 CI 日志中的警告。
- Harbor 的异步沙箱客户端绑定到一个事件循环，因此每个环境在自己的循环上同步驱动启动/执行/验证；因此，沙箱配置在整个生成批次中是连续的（像`e2b`这样的云后端可以减轻每个沙箱的成本）。
- 单个`HarborSpec`涵盖一套任务套件+一套安全带；尚不支持多套件训练。

## 参考

- [Harbor framework](https://www.harborframework.com)
- [Harbor RL training docs](https://www.harborframework.com/docs/training-workflows/rl)

### A2PO
https://huggingface.co/docs/trl/v1.9.2/a2po_trainer.md