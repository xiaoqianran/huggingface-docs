<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 实验追踪器

## GeneralTracker[[accelerate.tracking.GeneralTracker]]

####加速.tracking.GeneralTracker[[accelerate.tracking.GeneralTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L101)

用于所有日志集成实现的基本 Tracker 类。

每个函数都应该接受 `**kwargs` ，它将自动从提供给的基本字典传入
[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)。

应实现 `name`、`requires_logging_directory` 和 `tracker` 属性，以便：

`name` (`str`)：跟踪器类名的字符串表示，例如“TensorBoard” `requires_logging_directory`
(`bool`)：记录器是否需要一个目录来存储其日志。 `tracker` (`object`)：应该返回内部
跟踪器类使用的跟踪机制（例如 wandb 的 `run`）

实现还可以包含一个 `main_process_only` (`bool`) 属性来切换相关日志记录、初始化和
其他功能应该发生在主进程或跨所有进程（默认情况下将使用`True`）

finishaccelerate.tracking.GeneralTracker.finishhttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L171[]应运行跟踪 API 中的任何最终确定函数。如果 API 不应该有，那就不要
覆盖该方法。
#### 日志[[accelerate.tracking.GeneralTracker.log]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L159)

将 `values` 记录到当前运行。跟踪 API 的基本 `log` 实现应该放在这里，以及
`step 参数的特殊行为。

**参数：**

值（字典`str`到`str`、`float`或`int`）：要记录为键值对的值。这些值需要具有类型 `str`、`float` 或 `int`。

步骤（`int`，*可选*）：运行步骤。如果包含，日志将附属于此步骤。
####开始[[accelerate.tracking.GeneralTracker.start]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L142)

Accelerator 内跟踪器的延迟初始化以避免之前初始化 PartialState
InitProcessGroupKwargs。
#### store_init_configuration[[accelerate.tracking.GeneralTracker.store_init_configuration]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L148)

将 `values` 记录为运行的超参数。实现应使用实验配置
跟踪 API 的功能。

**参数：**值（字典`str`到`bool`、`str`、`float`或`int`）：要作为初始超参数存储为键值对的值。这些值需要具有类型 `bool`、`str`、`float`、`int` 或 `None`。

## TensorBoardTracker[[accelerate.tracking.TensorBoardTracker]]

####加速.tracking.TensorBoardTracker[[accelerate.tracking.TensorBoardTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L178)

支持`tensorboard`的`Tracker`类。应在脚本开始时初始化。

__init__accelerate.tracking.TensorBoardTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L194[{"name": "run_name", "val": ": str"}, {"name": "logging_dir", "val": ": Typing.Union[str, os.PathLike]"}, {"name": "**kwargs", "val": ""}]

**参数：**

run_name (`str`) : 实验运行的名称

logging_dir (`str`, `os.PathLike`) ：TensorBoard 日志的存储位置。

- ****kwargs** （附加关键字参数，*可选*）：传递给 `tensorboard.SummaryWriter.__init__` 方法的附加关键字参数。

## WandBTracker[[accelerate.tracking.WandBTracker]]

#### 加速.tracking.WandBTracker[[加速.tracking.WandBTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L293)支持 `wandb` 的 `Tracker` 类。应在脚本开始时初始化。

__init__accelerate.tracking.WandBTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L308[{"name": "run_name", "val": ": str"}, {"name": "**kwargs", "val": ""}]

**参数：**

run_name (`str`) ：实验运行的名称。

- ****kwargs** （附加关键字参数，*可选*）：传递给 `wandb.init` 方法的附加关键字参数。

## Trackio[[accelerate.tracking.TrackioTracker]]

#### 加速.tracking.TrackioTracker[[加速.tracking.TrackioTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L418)

支持 `trackio` 的 `Tracker` 类。应在脚本开始时初始化。

__init__accelerate.tracking.TrackioTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L435[{"name": "run_name", "val": ": str"}, {"name": "**kwargs", "val": ""}]

**参数：**

run_name (`str`) ：实验运行的名称。实例化 trackio 时将用作 `project` 名称。- ****kwargs** （附加关键字参数，*可选*）：传递给 `trackio.init` 方法的附加关键字参数。请参阅此[init](https://github.com/gradio-app/trackio/blob/814809552310468b13f84f33764f1369b4e5136c/trackio/__init__.py#L22)查看所有支持的关键字参数。

## CometMLTracker[[accelerate.tracking.CometMLTracker]]

####加速.tracking.CometMLTracker[[accelerate.tracking.CometMLTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L495)

支持`comet_ml`的`Tracker`类。应在脚本开始时初始化。

API 密钥必须存储在 Comet 配置文件中。

注意：
对于 `comet_ml` 版本 

__init__accelerate.tracking.CometMLTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L516[{"name": "run_name", "val": ": str"}, {"name": "**kwargs", "val": ""}]

**参数：**

run_name (`str`) ：实验运行的名称。

- ****kwargs** （附加关键字参数，*可选*）：传递给 `comet_ml.start` 方法的附加关键字参数：https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/start/

## AimTracker[[accelerate.tracking.AimTracker]]

#### 加速.tracking.AimTracker[[加速.tracking.AimTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L589)

支持`aim`的`Tracker`类。应在脚本开始时初始化。__init__accelerate.tracking.AimTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L603[{"name": "run_name", "val": ": str"}, {"name": "logging_dir", "val": ": Typing.Union[str, os.PathLike, NoneType] = '.'"}, {"name": "**kwargs", "val": ""}]

**参数：**

run_name (`str`) ：实验运行的名称。

- ****kwargs** （附加关键字参数，*可选*）：传递给 `Run.__init__` 方法的附加关键字参数。

## MLflowTracker[[accelerate.tracking.MLflowTracker]]

####加速.tracking.MLflowTracker[[accelerate.tracking.MLflowTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L692)

支持`mlflow`的`Tracker`类。应在脚本开始时初始化。__init__accelerate.tracking.MLflowTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L723[{"name": "experiment_name", "val": ": Typing.Optional[str] = None"}, {"name": "logging_dir", "val": ": Typing.Union[str, os.PathLike, NoneType] = None"}, {"name": "run_id", "val": ": Typing.Optional[str] = None"}, {"name": "tags", "val": ": Typing.Union[dict[str, Typing.Any], str, NoneType] = None"}, {"name": "nested_run", "val": ": Typing.Optional[bool] = False"}, {“name”：“run_name”，“val”：“：typing.Optional[str] = None”}，{“name”：“description”，“val”：“：typing.Optional[str] = None”}]

**参数：**

实验名称（`str`，*可选*）：实验的名称。环境变量 MLFLOW_EXPERIMENT_NAME 优先于该参数。

logging_dir （`str` 或 `os.PathLike`，默认为 `"."`）：mlflow 日志的存储位置。run_id (`str`, *可选*) ：如果指定，则获取具有指定 UUID 的运行以及该运行下的日志参数和指标。运行的结束时间未设置，其状态设置为正在运行，但运行的其他属性（source_version、source_type 等）不会更改。环境变量 MLFLOW_RUN_ID 优先于该参数。

标签（`Dict[str, str]`，*可选*）：可选的`str`键和值的`dict`，或来自`dict`的`str`转储，以在运行时设置为标签。如果正在恢复运行，则会在恢复的运行上设置这些标签。如果正在创建新运行，则会在新运行上设置这些标签。环境变量 MLFLOW_TAGS 的优先级高于该参数。

nested_run（`bool`，*可选*，默认为`False`）：控制运行是否嵌套在父运行中。 True 创建嵌套运行。环境变量 MLFLOW_NESTED_RUN 的优先级高于该参数。

run_name (`str`, *可选*) ：新运行的名称（存储为 mlflow.runName 标签）。仅当未指定 `run_id` 时使用。描述（`str`，*可选*）：填充运行描述框的可选字符串。如果正在恢复运行，则会在已恢复的运行上设置说明。如果正在创建新运行，则会在新运行上设置说明。

## ClearMLTracker[[accelerate.tracking.ClearMLTracker]]

####加速.tracking.ClearMLTracker[[accelerate.tracking.ClearMLTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L901)

支持`clearml`的`Tracker`类。应在脚本开始时初始化。

__init__accelerate.tracking.ClearMLTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L916[{"name": "run_name", "val": ": Typing.Optional[str] = None"}, {"name": "**kwargs", "val": ""}]

**参数：**

run_name (`str`, *可选*) ：实验的名称。环境变量 `CLEARML_PROJECT` 和 `CLEARML_TASK` 优先于该参数。

- ****kwargs** （附加关键字参数，*可选*）：Kwargs 传递给 `Task.__init__` 方法。

## SwanLabTracker[[accelerate.tracking.SwanLabTracker]]

#### 加速.tracking.SwanLabTracker[[加速.tracking.SwanLabTracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L1147)

支持`swanlab`的`Tracker`类。应在脚本开始时初始化。__init__accelerate.tracking.SwanLabTracker.__init__https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/tracking.py#L1162[{"name": "run_name", "val": ": str"}, {"name": "**kwargs", "val": ""}]

**参数：**

run_name (`str`) ：实验运行的名称。

- ****kwargs** （附加关键字参数，*可选*）：传递给 `swanlab.init` 方法的附加关键字参数。

### 有状态类
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/state.md