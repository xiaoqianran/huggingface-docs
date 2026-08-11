<!-- huggingface-docs: machine-translated zh-CN from English source -->

#PEFT

🤗 PEFT（参数高效微调）是一个库，用于有效地将大型预训练模型适应各种下游应用程序，而无需微调模型的所有参数，因为它的成本过高。 PEFT 方法仅微调少量（额外）模型参数 - 显着降低计算和存储成本 - 同时产生与完全微调模型相当的性能。这使得在消费类硬件上训练和存储大型语言模型 (LLM) 和其他大型模型变得更加容易。

PEFT 与 Transformers、Diffusers 和 Accelerate 库集成，提供更快、更简单的方法来加载、训练和使用大型模型进行推理。

  
    有许多方法可以“适应”现有模型，通常广泛地集成到模型中。 PEFT 可以被认为是任意模型适应方法（修改权重、包装层、操作 KV 缓存等）的框架，同时也可以作为许多微调方法的参考实现。<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="quicktour"
      >快速游览
      如果您是 🤗 PEFT 的新手，请从这里开始，了解该库的主要功能以及如何使用 PEFT 方法训练模型。
    
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./methods/overview"
      >方法概述
      了解不同类别的 PEFT 方法，以了解如何在您的模型中使用这些方法。

### 安装
https://huggingface.co/docs/peft/v0.20.0/install.md