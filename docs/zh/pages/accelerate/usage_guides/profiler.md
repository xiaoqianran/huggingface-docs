<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 分析器

Profiler 是一种允许在训练和推理期间收集性能指标的工具。 Profiler 的上下文管理器 API 可用于更好地了解哪些模型运算符最昂贵、检查其输入形状和堆栈跟踪、研究设备内核活动以及可视化执行跟踪。它可以深入了解模型的性能，使您能够优化和改进模型。

本指南介绍了如何使用 PyTorch Profiler 来测量模型运算符的时间和内存消耗以及如何将其与 Accelerate 集成。我们将介绍各种用例并为每个用例提供示例。

## 使用profiler分析执行时间

Profiler 允许检查在执行由探查器上下文管理器包装的代码范围期间调用了哪些运算符。

让我们看看如何使用分析器来分析执行时间：

```python
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

```python
from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.models as models

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True
)

accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
model = accelerator.prepare(model)

with accelerator.profile() as prof:
    with torch.no_grad():
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

结果表输出（省略一些列）：

```
---------------------------------  ------------  ------------  ------------  ------------  
                             Name      Self CPU     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  
                     aten::conv2d     171.000us      52.260ms       2.613ms            20  
                aten::convolution     227.000us      52.089ms       2.604ms            20  
               aten::_convolution     270.000us      51.862ms       2.593ms            20  
         aten::mkldnn_convolution      51.273ms      51.592ms       2.580ms            20  
                 aten::batch_norm     118.000us       7.059ms     352.950us            20  
     aten::_batch_norm_impl_index     315.000us       6.941ms     347.050us            20  
          aten::native_batch_norm       6.305ms       6.599ms     329.950us            20  
                 aten::max_pool2d      40.000us       4.008ms       4.008ms             1  
    aten::max_pool2d_with_indices       3.968ms       3.968ms       3.968ms             1  
                       aten::add_     780.000us     780.000us      27.857us            28  
---------------------------------  ------------  ------------  ------------  ------------  
Self CPU time total: 67.016ms
```

要获得更精细的结果粒度并包含操作员输入形状，请传递 `group_by_input_shape=True` （注意：这需要使用 `record_shapes=True` 运行分析器）：

```python
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
```

## 使用profiler分析内存消耗Profiler 还可以显示在模型运算符执行期间分配（或释放）的内存量（由模型张量使用）。要启用内存分析功能，请通过`profile_memory=True`。

```python
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

```python
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    profile_memory=True,
    record_shapes=True
)

accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
model = accelerator.prepare(model)

with accelerator.profile() as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

结果表输出（省略一些列）：

```
---------------------------------  ------------  ------------  ------------  
                             Name       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  
                      aten::empty      94.85 Mb      94.85 Mb           205  
    aten::max_pool2d_with_indices      11.48 Mb      11.48 Mb             1  
                      aten::addmm      19.53 Kb      19.53 Kb             1  
                       aten::mean      10.00 Kb      10.00 Kb             1  
              aten::empty_strided         492 b         492 b             5  
                        aten::cat         240 b         240 b             6  
                        aten::abs         480 b         240 b             4  
              aten::masked_select         120 b         112 b             1  
                         aten::ne          61 b          53 b             3  
                         aten::eq          30 b          30 b             1  
---------------------------------  ------------  ------------  ------------  
Self CPU time total: 69.332ms
```

如果您需要一两个聚合数字来进行实验跟踪而不是完整的运算符表，请将分析器与 PyTorch 公开的运行时内存统计信息结合起来。这通常足以记录每次运行的峰值内存使用情况：

```python
import torch
from accelerate import Accelerator, ProfileKwargs

def summarize_memory_stats():
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    profile_memory=True,
    record_shapes=True,
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

torch.cuda.reset_peak_memory_stats()

with accelerator.profile() as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
print(summarize_memory_stats())
```

此模式使探查器输出可用于调试，同时还生成可记录到实验跟踪器的紧凑值。

## 导出镀铬痕迹

您可以在 Chrome 跟踪查看器 (`chrome://tracing`) 中检查分析运算符和 CUDA 内核的序列：

![profile_export](https://github.com/huggingface/accelerate/assets/100389977/5acb193f-6d11-4f7b-9873-c600c19e8172)

```python
model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")
```

```python
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224).cuda()
profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    output_trace_dir="trace"
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
model = accelerator.prepare(model)

with accelerator.profile() as prof:
    model(inputs)

# The trace will be saved to the specified directory
```
对于其他硬件加速器，例如XPU，您可以将上面示例代码中的`cuda`更改为`xpu`。

## 使用 Profiler 分析长时间运行的作业

Profiler 提供了一个额外的 API 来处理长时间运行的作业（例如训练循环）。跟踪所有执行可能会很慢并且会产生非常大的跟踪文件。为了避免这种情况，请使用可选参数：- `schedule_option`：计划选项允许您控制分析何时处于活动状态。这对于长时间运行的作业非常有用，可以避免收集太多数据。可用键有 `wait`、`warmup`、`active`、`repeat` 和 `skip_first`。分析器将跳过前 `skip_first` 步骤，然后等待 `wait` 步骤，然后为接下来的 `warmup` 步骤进行预热，然后为接下来的 `active` 步骤进行活动记录，然后从 `wait` 步骤开始重复循环。可选的周期数由`repeat`参数指定，零值意味着周期将继续，直到分析完成。
- `on_trace_ready`：指定一个函数，它将对探查器的引用作为输入，并在每次新跟踪准备就绪时由探查器调用。

为了说明 API 的工作原理，请考虑以下示例：

```python
from torch.profiler import schedule

my_schedule = schedule(
    skip_first=1,
    wait=5,
    warmup=1,
    active=3,
    repeat=2
)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    on_trace_ready=trace_handler
) as p:
    for idx in range(8):
        model(inputs)
        p.step()
```

```python
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    schedule_option={"wait": 5, "warmup": 1, "active": 3, "repeat": 2, "skip_first": 1},
    on_trace_ready=trace_handler
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
model = accelerator.prepare(model)

with accelerator.profile() as prof:
    for idx in range(8):
        model(inputs)
        prof.step()
```

## 失败次数

使用公式来估计特定运算符（矩阵乘法和 2D 卷积）的 FLOP（浮点运算）。

测量浮点运算 (FLOPS)：

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_flops=True
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="flops", row_limit=10))
```

```python
profile_kwargs = ProfileKwargs(
    with_flops=True
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

with accelerator.profile() as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="flops", row_limit=10))
```

结果表输出（省略一些列）：

```
-------------------------------------------------------  ------------  ------------  ------------  
                                                   Name      Self CPU     Self CUDA    Total FLOPs  
-------------------------------------------------------  ------------  ------------  ------------  
                                           aten::conv2d     197.000us       0.000us  18135613440.000  
                                            aten::addmm     103.000us      17.000us     5120000.000  
                                              aten::mul      29.000us       2.000us          30.000  
                                      aten::convolution     409.000us       0.000us            --  
                                     aten::_convolution     253.000us       0.000us            --  
                                aten::cudnn_convolution       5.465ms       2.970ms            --  
                                        cudaEventRecord     138.000us       0.000us            --  
                                  cudaStreamIsCapturing      43.000us       0.000us            --  
                                  cudaStreamGetPriority      40.000us       0.000us            --  
                       cudaDeviceGetStreamPriorityRange      10.000us       0.000us            --  
-------------------------------------------------------  ------------  ------------  ------------  
Self CPU time total: 21.938ms
Self CUDA time total: 4.165ms
```

## 结论和更多信息PyTorch Profiler 是一个用于分析模型性能的强大工具。通过将其与 Accelerate 集成，您可以轻松分析模型并深入了解其性能，从而帮助您优化和改进它们。

更详细的信息请参阅[PyTorch Profiler documentation](https://pytorch.org/docs/stable/profiler.html)。

###执行过程
https://huggingface.co/docs/accelerate/v1.14.0/basic_tutorials/execution.md