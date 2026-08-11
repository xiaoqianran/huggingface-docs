<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 启动器

用于启动分布式流程培训的功能。

## 笔记本启动器[[accelerate.notebook_launcher]]

####加速.notebook_launcher[[accelerate.notebook_launcher]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/launchers.py#L43)

启动训练功能，如果当前环境可能的话，使用多个进程或多个节点
（例如具有多个核心的 TPU）。

要使用此功能，在调用之前必须在笔记本会话中对设备进行绝对零调用。如果有的话
完成后，您需要重新启动笔记本电脑并确保没有单元使用任何设备功能。

在您的环境中设置`ACCELERATE_DEBUG_MODE="1"`将在真正启动之前运行测试，以确保没有
已拨打了其中的电话。

示例：

```python
# Assume this is defined in a Jupyter Notebook on an instance with two devices
from accelerate import notebook_launcher

def train(*args):
    # Your training function here
    ...

notebook_launcher(train, args=(arg1, arg2), num_processes=2, mixed_precision="fp16")
```

**参数：**

function (`Callable`) ：要执行的训练函数。如果它接受参数，第一个参数应该是进程运行的索引。

args (`Tuple`) ：传递给函数的参数元组（它将接收`*args`）。

num_processes (`int`, *可选*) ：用于训练的进程数。如果 TPU 可用，Colab/Kaggle 中将默认为 8，否则为可用设备数量。混合精度（`str`，*可选*，默认为`"no"`）：如果`fp16`或`bf16`，将在多设备上使用混合精度训练。

use_port（`str`，*可选*，默认为`"29500"`）：启动多设备训练时用于在进程之间进行通信的端口。

master_addr (`str`，*可选*，默认为`"127.0.0.1"`)：用于进程之间通信的地址。

node_rank (`int`，*可选*，默认为 0) ：当前节点的排名。

num_nodes (`int`，*可选*，默认为 1) ：用于训练的节点数。

rdzv_backend (`str`，*可选*，默认为`"static"`)：要使用的集合方法，例如 'static' （默认）或 'c10d'

rdzv_endpoint（`str`，*可选*，默认为`""`）：rdzv 同步的端点。贮存。

rdzv_conf（`Dict`，*可选*，默认为`None`）：附加集合点配置。

rdzv_id (`str`，*可选*，默认为`"none"`)：作业的唯一运行 ID。

max_restarts (`int`，*可选*，默认为 0) ：弹性代理在失败之前对工作人员进行的最大重新启动次数。Monitor_interval (`float`，*可选*，默认为 0.1) ：elastic_agent 使用的时间间隔（以秒为单位）作为监视工作人员的周期。

log_line_prefix_template (`str`，*可选*，默认为`None`)：弹性启动日志记录的前缀模板。从 PyTorch 2.2.0 开始可用。

## debug_launcher[[accelerate.debug_launcher]]

####加速.debug_launcher[[accelerate.debug_launcher]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/launchers.py#L287)

使用 CPU 上的多个进程启动训练功能以进行调试。

此功能用于内部测试和调试，但不适用于真正的培训。它将
只使用CPU。

**参数：**

function (`Callable`) ：要执行的训练函数。

args (`Tuple`) ：传递给函数的参数元组（它将接收`*args`）。

num_processes (`int`，*可选*，默认为 2) ：用于训练的进程数。

### 日志记录[[accelerate.logging.get_logger]]
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/logging.md