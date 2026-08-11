<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在 Intel CPU 上进行训练

## CPU 训练优化的工作原理

Accelerate 完全支持 Intel CPU，您只需通过配置启用它即可。

**场景1**：无分布式CPU训练加速

在您的计算机上运行加速配置：

```bash
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: NO
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
```
这将生成一个配置文件，该文件将自动用于正确设置
执行时的默认选项

```bash
accelerate launch my_script.py --args_to_my_script
```

例如，以下是如何使用由 `accelerate config` 生成的 `default_config.yaml` 运行 NLP 示例 `examples/nlp_example.py` （从存储库的根目录）

```bash
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
```
```bash
accelerate launch examples/nlp_example.py
```

> [!警告]
> `accelerator.prepare` 目前只能处理同时准备多个模型（并且没有优化器）或单个模型优化器对进行训练。其他尝试（例如，两个模型优化器对）将引发详细错误。要解决此限制，请考虑对每个模型优化器对单独使用`accelerator.prepare`。

**场景2**：分布式CPU训练加速
我们使用英特尔 oneCCL 进行通信，并与英特尔® MPI 库相结合，在英特尔® 架构上提供灵活、高效、可扩展的集群消息传递。安装指南可参考[here](https://huggingface.co/docs/transformers/perf_train_cpu_many)

在你的机器（node0）上运行加速配置：```bash
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-CPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 4
-----------------------------------------------------------------------------------------------------------------------------------------------------------
What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 36.112.23.24
What is the port you will use to communicate with the main process? 29500
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: yes
Do you want accelerate to launch mpirun? [yes/NO]: yes
Please enter the path to the hostfile to use with mpirun [~/hostfile]: ~/hostfile
Enter the number of oneCCL worker threads [1]: 1
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
How many processes should be used for distributed training? [1]:16
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
```
例如，以下是如何运行 NLP 示例 `examples/nlp_example.py`（来自存储库的根）进行分布式 CPU 训练。

`default_config.yaml` 由 `accelerate config` 生成
```bash
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_CPU
downcast_bf16: 'no'
machine_rank: 0
main_process_ip: 36.112.23.24
main_process_port: 29500
main_training_function: main
mixed_precision: bf16
mpirun_config:
  mpirun_hostfile: /home/user/hostfile
num_machines: 4
num_processes: 16
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
```

设置以下环境并使用 intel MPI 启动训练

在 `node0` 中，您需要创建一个包含每个节点的 IP 地址的配置文件（例如主机文件），并将该配置文件路径作为参数传递。

如果您选择让 Accelerate 启动 `mpirun`，请确保主机文件的位置与配置中的路径匹配。

```bash
$ cat hostfile
xxx.xxx.xxx.xxx #node0 ip
xxx.xxx.xxx.xxx #node1 ip
xxx.xxx.xxx.xxx #node2 ip
xxx.xxx.xxx.xxx #node3 ip
```

```bash
accelerate launch examples/nlp_example.py
```

您也可以直接使用`mpirun`命令启动分布式训练，您需要在node0中运行以下命令，并且**16DDP**将在node0,node1,node2,node3中启用，BF16混合精度。使用此方法时，python 脚本、python 环境和加速配置文件需要在用于多 CPU 训练的所有机器上可用。

```bash
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
mpirun -f hostfile -n 16 -ppn 4 accelerate launch examples/nlp_example.py
```

### 低精度训练方法
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/low_ precision_training.md