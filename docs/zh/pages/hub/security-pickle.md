<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 泡菜扫描

Pickle 是机器学习中广泛使用的序列化格式。最值得注意的是，它是 PyTorch 模型权重的默认格式。

加载 pickle 文件时可能会发生危险的任意代码执行攻击。我们建议从您信任的用户和组织加载模型，依赖于签名提交，和/或使用 `from_tf=True` 自动转换机制从 TF 或 Jax 格式加载模型。我们还通过直接在集线器上显示/“审查”任何腌制文件中的导入列表来缓解此问题。最后，我们正在尝试一种新的、简单的权重序列化格式，称为 [⟦T10⟧](https://github.com/huggingface/safetensors)。

##什么是泡菜？

来自[official docs](https://docs.python.org/3/library/pickle.html)：

> `pickle` 模块实现了用于序列化和反序列化 Python 对象结构的二进制协议。

这意味着 pickle 是一种序列化协议，您可以使用它在各方之间有效地共享数据。

我们将 pickle 时生成的二进制文件称为 pickle。从本质上讲，pickle 基本上是一堆指令或操作码。正如您可能已经猜到的那样，它不是人类可读的。操作码在酸洗时生成，并在取消酸洗时按顺序读取。根据操作码，执行给定的操作。

这是一个小例子：

```python
import pickle
import pickletools

var = "data I want to share with a friend"

# store the pickle data in a file named 'payload.pkl'
with open('payload.pkl', 'wb') as f:
    pickle.dump(var, f)

# disassemble the pickle
# and print the instructions to the command line
with open('payload.pkl', 'rb') as f:
    pickletools.dis(f)
```

当您运行此命令时，它将创建一个 pickle 文件并在终端中打印以下说明：

```python
    0: \x80 PROTO      4
    2: \x95 FRAME      48
   11: \x8c SHORT_BINUNICODE 'data I want to share with a friend'
   57: \x94 MEMOIZE    (as 0)
   58: .    STOP
highest protocol among opcodes = 4
```

现在不用太担心说明，只要知道 [pickletools](https://docs.python.org/3/library/pickletools.html) 模块对于分析泡菜非常有用即可。它允许您读取文件中的指令，***无需***执行任何代码。

Pickle 不仅仅是一个序列化协议，它还允许用户在反序列化时运行 python 代码，从而提供更大的灵活性。听起来不太好，是吗？

## 为什么它很危险？

正如我们上面所说，反序列化 pickle 意味着代码可以被执行。但这有一定的局限性：您只能引用顶层模块中的函数和类；您不能将它们嵌入到 pickle 文件本身中。

回到绘图板：

```python
import pickle
import pickletools

class Data:
    def __init__(self, important_stuff: str):
        self.important_stuff = important_stuff

d = Data("42")

with open('payload.pkl', 'wb') as f:
    pickle.dump(d, f)
```

当我们运行这个脚本时，我们再次得到`payload.pkl`。当我们检查文件的内容时：

```bash

# cat payload.pkl
__main__Data)}important_stuff42sb.%

# hexyl payload.pkl
┌────────┬─────────────────────────┬─────────────────────────┬────────┬────────┐
│00000000│ 80 04 95 33 00 00 00 00 ┊ 00 00 00 8c 08 5f 5f 6d │×•×30000┊000×•__m│
│00000010│ 61 69 6e 5f 5f 94 8c 04 ┊ 44 61 74 61 94 93 94 29 │ain__××•┊Data×××)│
│00000020│ 81 94 7d 94 8c 0f 69 6d ┊ 70 6f 72 74 61 6e 74 5f │××}××•im┊portant_│
│00000030│ 73 74 75 66 66 94 8c 02 ┊ 34 32 94 73 62 2e       │stuff××•┊42×sb.  │
└────────┴─────────────────────────┴─────────────────────────┴────────┴────────┘
```我们可以看到里面没有太多内容，只有一些操作码和相关数据。你可能会想，那么泡菜有什么问题呢？

让我们尝试一下其他的东西：

```python
from fickling.pickle import Pickled
import pickle

# Create a malicious pickle
data = "my friend needs to know this"

pickle_bin = pickle.dumps(data)

p = Pickled.load(pickle_bin)

p.insert_python_exec('print("you\'ve been pwned !")')

with open('payload.pkl', 'wb') as f:
    p.dump(f)

# innocently unpickle and get your friend's data
with open('payload.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)
```

为了简单起见，这里我们使用 [fickling](https://github.com/trailofbits/fickling) 库。它允许我们添加 pickle 指令来通过 `exec` 函数执行字符串中包含的代码。这就是如何规避无法在 pickles 中定义函数或类这一事实的方法：对保存为字符串的 python 代码运行 exec。

当您运行它时，它会创建一个 `payload.pkl` 并打印以下内容：

```
you've been pwned !
my friend needs to know this
```

如果我们检查 pickle 文件的内容，我们会得到：

```bash
# cat payload.pkl
c__builtin__
exec
(Vprint("you've been pwned !")
tR my friend needs to know this.%

# hexyl payload.pkl
┌────────┬─────────────────────────┬─────────────────────────┬────────┬────────┐
│00000000│ 63 5f 5f 62 75 69 6c 74 ┊ 69 6e 5f 5f 0a 65 78 65 │c__built┊in___exe│
│00000010│ 63 0a 28 56 70 72 69 6e ┊ 74 28 22 79 6f 75 27 76 │c_(Vprin┊t("you'v│
│00000020│ 65 20 62 65 65 6e 20 70 ┊ 77 6e 65 64 20 21 22 29 │e been p┊wned !")│
│00000030│ 0a 74 52 80 04 95 20 00 ┊ 00 00 00 00 00 00 8c 1c │_tR×•× 0┊000000×•│
│00000040│ 6d 79 20 66 72 69 65 6e ┊ 64 20 6e 65 65 64 73 20 │my frien┊d needs │
│00000050│ 74 6f 20 6b 6e 6f 77 20 ┊ 74 68 69 73 94 2e       │to know ┊this×.  │
└────────┴─────────────────────────┴─────────────────────────┴────────┴────────┘
```

基本上，这就是你 unpickle 时发生的情况：

```python
# ...
opcodes_stack = [exec_func, "malicious argument", "REDUCE"]
opcode = stack.pop()
if opcode == "REDUCE":
    arg = opcodes_stack.pop()
    callable = opcodes_stack.pop()
    opcodes_stack.append(callable(arg))
# ...
```

构成威胁的指令是`STACK_GLOBAL`、`GLOBAL`和`REDUCE`。

`REDUCE` 告诉 unpickler 使用提供的参数执行函数，`*GLOBAL` 指令告诉 unpickler `import` 的东西。

总而言之，泡菜很危险，因为：

- 导入python模块时，可以执行任意代码
- 您可以导入诸如`eval`或`exec`之类的内置函数，它们可用于执行任意代码
- 实例化对象时，可能会调用构造函数这就是为什么在大多数文档中都声明使用pickle，不要从不受信任的来源取消pickle数据。

## 缓解策略

***不要使用泡菜***

Luc 的建议是合理的，但 pickle 被广泛使用，并且不会很快消失：找到一种每个人都满意的新格式并启动更改需要一些时间。

那么我们现在能做什么呢？

### 从您信任的用户和组织加载文件

在 Hub 上，您有能力[sign your commits with a GPG key](./security-gpg)。这**不能**保证您的文件是安全的，但它确实保证了文件的来源。

如果您了解并信任用户 A，并且 Hub 上包含该文件的提交是由用户 A 的 GPG 密钥签名的，则可以非常安全地假设您可以信任该文件。

### 从 TF 或 Flax 加载模型权重

TensorFlow 和 Flax 检查点不受影响，并且可以使用 `from_tf` 和 `from_flax` kwargs 的 `from_pretrained` 方法加载到 PyTorch 架构中来规避此问题。

例如：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased", from_flax=True)
```

### 使用您自己的序列化格式

- [MsgPack](https://msgpack.org/index.html)
- [Protobuf](https://developers.google.com/protocol-buffers)
- [Cap'n'proto](https://capnproto.org/)
- [Avro](https://avro.apache.org/)
- [safetensors](https://github.com/huggingface/safetensors)

最后一种格式，`safetensors`，是我们目前正在研究和试验的一种简单的序列化格式！如果可以的话请帮忙或贡献🔥。

### 改进`torch.load/save`PyTorch 正在进行关于 [Safe way of loading only weights from *.pt file by default](https://github.com/pytorch/pytorch/issues/52181) 的公开讨论 - 请参与其中！

### Hub 的安全扫描器

#### 我们现在拥有什么

我们创建了一个安全扫描器，可以扫描推送到集线器的每个文件并运行安全检查。在撰写本文时，它运行两种类型的扫描：

- ClamAV 扫描
- Pickle 导入扫描

对于 ClamAV 扫描，文件通过开源防病毒软件 [ClamAV](https://www.clamav.net) 运行。虽然这涵盖了大量危险文件，但它不涵盖 pickle 漏洞。

我们实现了 Pickle 导入扫描，它提取 pickle 文件中引用的导入列表。每次您上传 `pytorch_model.bin` 或任何其他腌制文件时，都会运行此扫描。

在中心上，导入列表将显示在每个包含导入的文件旁边。如果任何导入看起来可疑，则会突出显示。 

我们通过 [⟦T29⟧](https://docs.python.org/3/library/pickletools.html#pickletools.genops) 获得这些数据，它允许我们在不执行潜在危险代码的情况下读取文件。

请注意，这可以让您知道在解封文件时，它是否会 `REDUCE` `*GLOBAL` 导入的潜在危险函数。***免责声明***：这并非100%万无一失。作为用户，您有责任检查某些内容是否安全。我们不会主动审核 python 包的安全性，我们会尽力维护安全/不安全的导入列表。
如果您认为某些内容不安全，请通过向我们的网站huggingface.co 发送电子邮件来标记该情况，并与我们联系

#### 潜在的解决方案

人们可以考虑创建一个自定义的 [Unpickler](https://docs.python.org/3/library/pickle.html#pickle.Unpickler)，例如 [this one](https://github.com/facebookresearch/CrypTen/blob/main/crypten/common/serial.py)。但正如我们在这个[sophisticated exploit](https://ctftime.org/writeup/16723)中看到的，这是行不通的。

值得庆幸的是，总是存在 `eval` 导入的痕迹，因此直接读取操作码应该可以捕获恶意使用。

我建议的当前解决方案是创建一个类似于 `.gitignore` 但用于导入的文件。

该文件将是导入的白名单，如果白名单中未包含导入，则会将 `pytorch_model.bin` 文件标记为危险。

人们可以想象有一种正则表达式式的格式，你可以通过像这样的简单行来允许所有 numpy 子模块：`numpy.*`。

## 进一步阅读

[pickle - Python object serialization - Python 3.10.6 documentation](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)

[Dangerous Pickles - Malicious Python Serialization](https://intoli.com/blog/dangerous-pickles/)

[GitHub - trailofbits/fickling: A Python pickling decompiler and static analyzer](https://github.com/trailofbits/fickling)

[Exploiting Python pickles](https://davidhamann.de/2020/04/05/exploiting-python-pickle/)

[cpython/pickletools.py at 3.10 · python/cpython](https://github.com/python/cpython/blob/3.10/Lib/pickletools.py)

[cpython/pickle.py at 3.10 · python/cpython](https://github.com/python/cpython/blob/3.10/Lib/pickle.py)

[CrypTen/serial.py at main · facebookresearch/CrypTen](https://github.com/facebookresearch/CrypTen/blob/main/crypten/common/serial.py)

[CTFtime.org / Balsn CTF 2019 / pyshv1 / Writeup](https://ctftime.org/writeup/16723)

[Rehabilitating Python's pickle module](https://github.com/moreati/pickle-fuzz)

### 资源组
https://huggingface.co/docs/hub/enterprise-resource-groups.md