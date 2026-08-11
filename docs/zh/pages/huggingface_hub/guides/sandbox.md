<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 沙箱

沙箱是一个独立的云机器，您可以在几秒钟内启动，使用实时流输出运行命令，并将文件移入和移出——所有这些都可以通过 Python 或 CLI 进行。沙箱构建在 [Jobs](./jobs) 之上：在底层，沙箱只是一个运行小型服务器的作业，该服务器通过 HTTP 公开命令执行和文件传输。

当您需要在自己的机器以外的地方运行代码时，它们非常适合：

- **运行不受信任或人工智能生成的代码** - 让代理执行任意代码而不授予其访问您的文件系统的权限。
- **可重复的构建和实验** — 在 CPU 或 GPU 上干净、定义良好的图像上运行。
- **分散工作**——以较低的成本启动数百个并行环境（RL 部署、评估、批处理工具执行）。

任何带有 `/bin/sh` 的 Docker 镜像都可以工作——不需要预先安装 Python、pip 或代理（在启动时注入一个小的静态服务器二进制文件）。

> [!提示]
> 好奇这在幕后是如何工作的——工作中的服务器、无状态身份验证，或者单个作业如何托管许多独立的沙箱？请参阅[Sandboxes conceptual guide](../concepts/sandbox)。

## 两种沙箱获取沙箱有两种方法。两者都给你相同的[Sandbox](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox)对象（相同的`run`，`files`，`connect`，`kill`）；它们的区别仅在于底层机器的分配方式：

|            | [Sandbox.create()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox.create) — **专用** | [SandboxPool](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxPool) — **共享/池** |
| ---------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
|地图|一项作业 = **一个沙箱**（整个虚拟机）|一项作业 = **许多沙箱**（一台虚拟机，已打包）|
|隔离 |完整虚拟机 | uid + [Landlock](https://docs.kernel.org/userspace-api/landlock.html)（同一用户信任）|
|冷启动|每个沙箱约 6 秒 |第一个主机约 6 秒，然后每个主机约 1 次往返 |
|成本|每个沙箱一个虚拟机 |每个**主机**一个虚拟机，分摊到许多沙箱 |
|图形处理器 | ✅ | ❌（仅限 CPU）||最适合 |单一沙箱、GPU 工作负载、不受信任的代码 |许多廉价的 CPU 沙箱（强化学习、扇出）|

经验法则：需要 GPU 或运行相互不信任的代码 → 专用。需要数百个廉价的 CPU 沙箱 → 一个池。

## 快速入门

```python
>>> from huggingface_hub import Sandbox

>>> with Sandbox.create() as sbx:                      # ready in ~6s
...     result = sbx.run("python -c 'print(40 + 2)'")  # ~100ms per command
...     print(result.stdout)
42
```

选择任意图像和硬件[flavor](./jobs#select-the-hardware)：

```python
>>> sbx = Sandbox.create(image="alpine:3.20")
>>> sbx = Sandbox.create(image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel", flavor="a10g-small")
```

## 运行命令

[Sandbox.run()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox.run) 执行命令并等待它。传递 shell 字符串或 argv 列表：

```python
>>> sbx.run("pip install -q numpy")                       # string  → runs through /bin/sh -c
>>> sbx.run(["python", "-c", "import numpy; print(numpy.__version__)"])  # list → exec'd directly

# Live output streaming, plus env, cwd, timeout, stdin
>>> sbx.run("make -j4", cwd="/app", env={"CC": "gcc"}, timeout=600, on_stdout=print, on_stderr=print)
```

默认情况下，模式是从类型推断的（字符串贯穿`/bin/sh -c`，列表直接作为argv执行）。通过 `shell=` 使其显式化 - 方便避免经典的 footgun，其中像 `["echo hi"]` 这样的单元素列表作为名为 `"echo hi"` 的单个程序执行：

```python
>>> sbx.run("echo $HOME && ls | wc -l", shell=True)   # force the shell (pipes, globs, $VARS)
>>> sbx.run(["git", "commit", "-m", msg], shell=False)  # force argv (no quoting surprises)
```

`shell=True` 需要一个字符串，`shell=False` 需要一个列表；传递错误的类型会引发`ValueError`。

退出非零的命令会引发 `SandboxCommandError`（附加有 `stdout`、`stderr` 和 `exit_code`）。通过 `check=False` 来取回 [SandboxCommandResult](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxCommandResult)，而不是加注：

```python
>>> result = sbx.run("test -f /tmp/missing", check=False)
>>> result.exit_code
1
```

### 后台进程

通过 `background=True` 启动一个长时间运行的进程（服务器、观察者、训练运行）而无需等待。 `run` 立即返回 [SandboxProcess](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxProcess) 而不是 [SandboxCommandResult](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxCommandResult)：

```python
>>> proc = sbx.run("python -m http.server 8000", background=True)
>>> proc
SandboxProcess(pid=1234, cmd='python -m http.server 8000', tag=None, started_at_ms=1700000000000, running=True, exit_code=None)
```列出沙箱的进程并在完成后停止其中一个。已完成的进程将保持列出状态（带有 `running=False` 及其 `exit_code`），直到沙箱被删除，因此您可以判断进程是否仍然存在或已退出：

```python
>>> sbx.processes()
[SandboxProcess(pid=1234, cmd='python -m http.server 8000', tag=None, started_at_ms=1700000000000, running=True, exit_code=None)]
>>> proc.kill()
```

流/仅等待选项（`timeout`、`stdin`、`on_stdout`、`on_stderr`、`check`）不适用于后台模式 — 仅支持`env`、`cwd` 和 `shell`。

## 文件

```python
>>> sbx.files.write("/app/script.py", "print('hi')")     # str | bytes | file-like
>>> sbx.files.read_text("/app/script.py")
"print('hi')"
>>> sbx.files.upload("local_data.csv", "/data/data.csv")  # local -> sandbox
>>> sbx.files.download("/data/results.bin", "results.bin")  # sandbox -> local
>>> sbx.files.list("/data")
[FileEntry(name='data.csv', path='/data/data.csv', type='file', size=5324, ...)]
```

其他助手：`stat`、`exists`、`mkdir`、`delete`。

## 到达沙箱内的服务器

在沙箱中（在后台）启动一个服务器，然后使用[Sandbox.proxy_url_for()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox.proxy_url_for)从外部访问它——请求由正在工作的沙箱服务器转发到您的内部服务器，因此没有额外的公共端口可以公开。它适用于纯 HTTP、服务器发送事件和 WebSocket。将 URL 与 `Sandbox.proxy_headers` 配对以进行身份验证（您的 WebSocket/HTTP 客户端必须发送它们）：

```python
>>> import httpx
>>> with Sandbox.create() as sbx:
...     sbx.files.write("app.py", "...")  # a server exposing e.g. /hello and /ws
...     sbx.run("uvicorn app:app --host 127.0.0.1 --port 8000", background=True)
...     # plain HTTP
...     r = httpx.get(sbx.proxy_url_for(8000, "/hello"), headers=sbx.proxy_headers)
...     # WebSocket: ask for a wss:// URL
...     ws_url = sbx.proxy_url_for(8000, "/ws", scheme="wss://")
```

内部服务器必须如何侦听取决于沙箱类型：

- **专用** ([Sandbox.create()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox.create))：在`127.0.0.1:<port>`上绑定普通TCP端口。
- **池/共享**（[SandboxPool](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxPool)）：池化沙箱无法绑定 TCP 端口（Landlock），因此在 `$SBX_PROXY_DIR/<port>.sock` 处侦听 **unix 套接字**（该环境变量在每个沙箱中设置），例如`uvicorn app:app --uds $SBX_PROXY_DIR/8000.sock`。无论哪种方式，客户端 (`proxy_url_for` / `proxy_headers`) 都是相同的。

## 生命周期沙箱比创建它的进程寿命更长——您可以现在创建它，然后从任何持有相同 HF 令牌的机器重新连接，无需复制任何状态：

```python
>>> sbx = Sandbox.create()
>>> sbx.id
'687f911eaea852de79c4a50a'

# Later, from anywhere:
>>> sbx = Sandbox.connect("687f911eaea852de79c4a50a")
>>> sbx.kill()           # terminate now
```

- `idle_timeout`（默认10分钟）是真正的守护者：一旦没有进行API调用并且没有进程正在运行，它就会关闭沙箱，因此废弃的沙箱将停止计费。在创建时设置它（`Sandbox.create(idle_timeout="30m")`）或传递`None`以禁用。
- 该作业还具有固定的 24 小时最长寿命作为硬后挡板（不可配置）。
- 除非您选择使用`forward_hf_token=True`，否则您的 HF 代币永远不会发送到沙箱中。

## 同时使用多个沙箱：SandboxPool

当您需要许多沙箱（并行 RL 部署、扇出评估、批处理工具执行）时，每个沙箱一个作业是浪费的：每个作业都需要支付完整的虚拟机冷启动费用，并为需要几 MB RAM 的工作负载容纳一整台机器。相反，[SandboxPool](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxPool) 将许多轻量级沙箱打包到几个共享主机作业中 - 一个计费虚拟机为数十个沙箱提供服务，因此每个沙箱的成本下降了该因素，并且每个沙箱的冷启动大约是一次网络往返。

```python
>>> from huggingface_hub import SandboxPool

>>> with SandboxPool(image="python:3.12", flavor="cpu-basic", warm_up=2) as pool:
...     boxes = [pool.create() for _ in range(100)]   # packed across the 2 warm host VMs
...     print(boxes[0].run("echo hi").stdout)          # each box is a normal Sandbox
hi
```每个`create()`返回一个完整的[Sandbox](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox)；反复调用它以扇出。池根据需要启动主机作业，为每个主机打包 `sandboxes_per_host` 沙箱，并终止 `close()` 上的所有内容（或者当主机空闲时，作为计费后备）。典型的扇出模式：

```python
>>> from concurrent.futures import ThreadPoolExecutor
>>> with SandboxPool(image="python:3.12", warm_up=4) as pool:
...     boxes = [pool.create() for _ in tasks]
...     with ThreadPoolExecutor(32) as ex:
...         outputs = list(ex.map(lambda b, t: b.run(t.cmd).stdout, boxes, tasks))
```

Env 和 `idle_timeout` 是每个沙箱的（它们属于 `create()`，而不是池），因此一个池中的沙箱可以有不同的环境：

```python
>>> sbx = pool.create(env={"SEED": "42"}, idle_timeout="5m", forward_hf_token=True)
```

池化沙箱是一个完整的 [Sandbox](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox)，但一些输入由主机固定，而不是每个沙箱 - 因此 [SandboxPool.create()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxPool.create) 接受比 [Sandbox.create()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox.create) 更小的参数集：

|输入 | `Sandbox.create`（专用）| `SandboxPool.create`（合并）|
| ------------------ | ---------------------------- | ---------------------------------------------------- |
| `image`、`flavor` |每个沙箱 |由池固定（在主机上设置一次）|
| `volumes` |每个沙箱 | **不可用**（在主机启动时安装）|
| `env` |每个沙箱 |每个沙箱 |
| `idle_timeout` |每个沙箱 |每个沙箱 |
| `forward_hf_token` |每个沙箱 |每个沙箱 |池化沙箱共享一个长期存在的主机作业，因此没有像专用沙箱那样的加密作业秘密通道。但是池化沙箱的 `env` 在沙箱创建时（而不是在作业启动时）传递到主机服务器，因此它永远不会存储在任何作业的元数据中 - 将可能的秘密作为普通的 `env` 传递。

### 按需生长和预热

`pool.create()` 一次创建一个沙箱，在启动新主机之前重用仍然具有可用容量的主机 - 因此您可以在工作到达时生成沙箱，并将它们打包到温暖的主机上：

```python
>>> pool = SandboxPool(image="python:3.12", flavor="cpu-basic")
>>> sbx = pool.create()    # boots the first host (~6s)
>>> sbx = pool.create()    # packs onto the same warm host (~one round-trip, no new VM)
```

为了避免主机在前几次调用时冷启动，请预先配置主机`warm_up=N`（在第一个`create()`启动）或通过预先调用`pool.warm(N)`。

### 跨进程和机器重用池

热主机是通过作业标签发现的，因此可以跨进程重用：具有相同 `image`/`flavor`/`name` 的全新`SandboxPool`附加到早期运行的主机上，而不是启动自己的主机。传递 `name=` 以保持单独的池与共享主机。要从没有本地状态的另一台计算机重新连接，请通过池 ID 与 [SandboxPool.connect()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.SandboxPool.connect) 重新连接 — 它会找到正在运行的主机，从该主机作业重建池的配置（图像、风味、包装密度），并准备好 `create()` 更多：

```python
>>> pool = SandboxPool.connect("pool-ae9f7efe0bc7")   # from anywhere, no config needed
>>> sbx = pool.create()
```

`connect()`'d 池不拥有共享主机（其他客户端可能正在使用它们），因此 - 就像 [Sandbox.connect()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox.connect) - 保留其 `with` 块（或调用 `close()`）只会释放本地 HTTP 客户端并使主机保持运行。使用 `pool delete` / `hf sandbox pool delete <id>` 显式终止池的主机。

> [!警告]
> 主机内的沙箱通过不同的 uid 以及每个沙箱的 Landlock 规则集相互隔离 - 它们无法读取、发送信号或写入彼此的文件，并且每个沙箱都被限制在自己的私人空间中。这是*一个用户自己的*并行工作负载的正确边界。对于相互敌对的不可信代码或 GPU，请使用[Sandbox.create()](/docs/huggingface_hub/v1.27.0/en/package_reference/sandbox#huggingface_hub.Sandbox.create)（每个沙箱一个单独的虚拟机）。 [conceptual guide](../concepts/sandbox#isolation-in-a-pool-uid--landlock) 中详细介绍了权衡。

## 从 CLI

`hf sandbox` 命令镜像 Python API。专用沙箱：

```bash
>>> hf sandbox create
✓ Sandbox ready id=687f911eaea852de79c4a50a image=python:3.12 elapsed=6.0s

>>> hf sandbox exec 687f911eaea852de79c4a50a -- python -c "print('hi')"
hi

>>> hf sandbox cp data.csv 687f911eaea852de79c4a50a:/data/data.csv
>>> hf sandbox kill 687f911eaea852de79c4a50a
```

`hf sandbox exec` 实时流式输出并以命令的退出代码退出，因此它由脚本组成：

```bash
hf sandbox exec $ID -- pytest && echo "tests passed"
```使用 `hf sandbox spawn` 在后台启动一个长时间运行的进程（打印其 pid），然后列出或停止进程。该列表显示每个进程的状态（`running`或`exited (<code>)`）：

```bash
>>> hf sandbox spawn $ID -- python -m http.server 8000
✓ Process started sandbox=687f... pid=1234

>>> hf sandbox process ls $ID
pid   status   cmd
1234  running  python -m http.server 8000

>>> hf sandbox process kill $ID 1234
```

对于许多廉价的共享沙箱，先预热一次池，然后按需创建：

```bash
# Warm a pool -> prints a pool id (billing starts: a host VM is now running)
>>> hf sandbox pool create python:3.12 --flavor cpu-basic
✓ Pool created id=pool-ae9f7efe0bc7 image=python:3.12 flavor=cpu-basic host=687f... elapsed=5.7s

# Each create packs onto a host with room (found by the pool id, from any machine);
# only when every host is full does it boot a duplicate. Env is per-sandbox (pooled
# sandboxes share a host, so there's no encrypted-secrets channel — use --env).
>>> hf sandbox create --pool pool-ae9f7efe0bc7 --env LOG_LEVEL=debug
>>> hf sandbox create --pool pool-ae9f7efe0bc7

>>> hf sandbox pool ls
>>> hf sandbox pool delete pool-ae9f7efe0bc7    # terminate the pool's hosts (and their sandboxes)
```

`hf sandbox create --pool` 产生共享沙箱；它的 id 看起来像 `<host_job_id>.<local_id>` 并且在专用 id 所做的任何地方都有效（`exec`、`cp`、`kill`）。池没有本地状态 - 它只是其正在运行的主机虚拟机，通过池 ID 找到 - 因此它可以在任何计算机上工作，并在其所有主机消失（被杀死或空闲超时）后停止存在。

### 与讨论和 Pull 请求互动
https://huggingface.co/docs/huggingface_hub/v1.27.0/guides/community.md