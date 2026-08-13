<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 访问基准排行榜数据

Hub上的[Benchmark datasets](./eval-results#benchmark-datasets)包含按评估分数排名的排行榜模型。您可以通过编程方式访问这些数据，以在其基础上进行分析、构建仪表板或工具。

## 发现官方基准

使用`huggingface_hub`查找所有官方基准数据集：

```python
from huggingface_hub import HfApi

api = HfApi()
for ds in api.list_datasets(benchmark=True):
    print(ds.id)
```

或者直接通过 REST API（对于代理和脚本编写很有用）：

```
GET https://huggingface.co/api/datasets?filter=benchmark:official
```

## 获取排行榜排名

排行榜 API 返回基准数据集的排名模型分数：

```
GET https://huggingface.co/api/datasets/{dataset_id}/leaderboard
```

使用 [⟦T12⟧](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.get_dataset_leaderboard) 获取排序模型分数作为键入的 [⟦T13⟧](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.DatasetLeaderboardEntry) 对象：

```python
from huggingface_hub import HfApi

api = HfApi()
leaderboard = api.get_dataset_leaderboard("SWE-bench/SWE-bench_Verified")

for entry in leaderboard[:5]:
    print(f"#{entry.rank} {entry.model_id}: {entry.value}")
```

> [!提示]
> `huggingface_hub` 默认使用您缓存的令牌。对于门控基准数据集，请确保您已登录 (`hf auth login`) 或显式传递令牌：
> ```python
> leaderboard = api.get_dataset_leaderboard("gated/benchmark", token="hf_...")
> ```

> [!提示]
> 卷曲一行以快速访问（对于代理和脚本编写很有用）：
> ```bash
> curl https://huggingface.co/api/datasets/cais/hle/leaderboard \
>   --header "Authorization: Bearer $(cat ~/.cache/huggingface/token)" | jq .
> ```

### 响应字段

每个[⟦T16⟧](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.DatasetLeaderboardEntry)包含：|领域|描述 |
|---|---|
| `rank` |排行榜上的位置 |
| `model_id` |完整型号 ID（例如 `Qwen/Qwen3.5-397B-A17B`）|
| `value` |基准分数|
| `verified` |结果是否经过独立验证 |
| `author` | [⟦T23⟧](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.User) 或 [⟦T24⟧](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.Organization) 对象 |
| `source` |结果从哪里提交（模型卡、外部等）|
| `filename` |评估结果 YAML 文件的路径（例如 `.eval_results/swe_bench_verified.yaml`）|
| `pull_request` |基准数据集存储库上提交的 PR 编号 |
| `notes` |与条目相关的可选注释 |

## 预先聚合的多基准数据集

如果您希望在单个文件中获得多个基准测试的分数，[⟦T30⟧](https://huggingface.co/datasets/OpenEvals/leaderboard-data) 数据集会将官方基准测试的分数汇总到一个 Parquet 文件中：

您可以使用 `hf://` 路径直接使用 [pandas](./datasets-pandas) 加载它：

```python
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/OpenEvals/leaderboard-data/data/train-00000-of-00001.parquet"
)
print(df[["model_name", "provider", "aime2026_score", "mmluPro_score"]].head())
```

这是无需调用多个 API 端点即可获取交叉基准视图的最快方法。

## 丰富模型元数据

使用 `huggingface_hub` 通过发布日期、参数计数和其他元数据来丰富排行榜数据：

```python
from huggingface_hub import HfApi

api = HfApi()
info = api.model_info("Qwen/Qwen3.5-397B-A17B")

print(f"Released: {info.created_at}")
print(f"Parameters: {info.safetensors.total / 1e9:.1f}B" if info.safetensors else "")
```

## 以模型为中心的视图：每个模型的评估结果排行榜 API 提供了一种以数据集为中心的视图（所有模型都基于一个基准）。相反，单个模型的所有基准分数 - 使用 `model_info` 和 `expand=["evalResults"]`：

```python
from huggingface_hub import HfApi

api = HfApi()
info = api.model_info("Qwen/Qwen3.5-397B-A17B", expand=["evalResults"])

for result in info.eval_results:
    print(f"{result.dataset_id}: {result.value}")
```

这将返回从模型的 `.eval_results/` 文件解析的 [⟦T35⟧](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.EvalResultEntry) 对象。

## 示例：基于排行榜数据构建

[Benchmark Leaderboard Race](https://huggingface.co/spaces/davanstrien/benchmark-race) Space 结合了这些数据源，创建了模型排名如何随时间演变的动画可视化。您可以在此数据的基础上构建自己的分析和可视化 - 有关完整示例，请参阅 [source code](https://huggingface.co/spaces/davanstrien/benchmark-race/tree/main)。

## 在网页中嵌入排行榜

您可以使用 iframe 将基准数据集的排行榜直接嵌入到您自己的网页中。

要使用的 URL 为 `https://huggingface.co/datasets/<namespace>/<dataset-name>/embed/leaderboard`，其中 `<namespace>` 是基准数据集的所有者（用户或组织），`<dataset-name>` 是数据集的名称。

```html
<iframe
  src="https://huggingface.co/datasets/cais/hle/embed/leaderboard"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>
```

### 参数

您可以通过在 iframe URL 中传递查询参数来配置嵌入式排行榜：|参数|描述 |
|---|---|
| `leaderboard_task_id` |要显示的任务的 ID，如基准测试的 `eval.yaml` 中所定义（例如 `gpqa_diamond`）。默认为第一个任务。 |
| `eval_result` |要在排行榜上突出显示的型号 ID（例如 `meta-llama/Llama-3.1-8B`）。 |
| `leaderboard_max_params` |按最大参数计数过滤行。接受以下值之一：`1B`、`3B`、`6B`、`12B`、`32B`、`128B` 或 `500B`。 |
| `leaderboard_is_expanded` |设置为 `true` 可使排行榜完全展开而不是折叠。 |

例如，要嵌入 HLE 排行榜，并展开表格并突出显示特定模型：

```html
<iframe
  src="https://huggingface.co/datasets/cais/hle/embed/leaderboard?leaderboard_is_expanded=true&eval_result=meta-llama/Llama-3.1-8B"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>
```

嵌入仅适用于有评估结果的[official benchmark datasets](./eval-results#benchmark-datasets)。

## 相关

- [Eval Results](./eval-results) — 如何提交评估结果并注册基准
- [Official Benchmark Datasets](https://huggingface.co/datasets?benchmark=benchmark:official&sort=trending) — 浏览所有官方基准测试

### 在抱脸处使用🧨 `diffusers`
https://huggingface.co/docs/hub/diffusers.md