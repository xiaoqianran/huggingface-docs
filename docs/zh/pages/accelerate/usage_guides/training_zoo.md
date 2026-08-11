<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 动物园示例

下面包含展示 Accelerate 的教程和脚本的非详尽列表。

## 官方加速示例：

### 基本示例

这些示例展示了 Accelerate 的基本功能，是一个很好的起点

- [Barebones NLP example](https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py)
- [Barebones distributed NLP example in a Jupyter Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)
- [Barebones computer vision example](https://github.com/huggingface/accelerate/blob/main/examples/cv_example.py)
- [Barebones distributed computer vision example in a Jupyter Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_cv_example.ipynb)
- [Using Accelerate in Kaggle](https://www.kaggle.com/code/muellerzr/multi-gpu-and-accelerate)

### 功能具体示例

这些示例展示了 Accelerate 框架提供的特定功能

- [Automatic memory-aware gradient accumulation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/automatic_gradient_accumulation.py)
- [Checkpointing states](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py)
- [Cross validation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/cross_validation.py)
- [DeepSpeed](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py)
- [Fully Sharded Data Parallelism](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/fsdp_with_peak_mem_tracking.py)
- [Gradient accumulation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/gradient_accumulation.py)
- [Memory-aware batch size finder](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/memory.py)
- [Metric Computation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py)
- [Using Trackers](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/tracking.py)
- [Using Megatron-LM](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/megatron_lm_gpt_pretraining.py)

### 完整示例 

这些示例展示了“功能特定示例”中显示的 Accelerate 中的每个功能

- [Complete NLP example](https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py)
- [Complete computer vision example](https://github.com/huggingface/accelerate/blob/main/examples/complete_cv_example.py)
- [Very complete and extensible vision example showcasing SLURM, hydra, and a very extensible usage of the framework](https://github.com/yuvalkirstain/PickScore)
- [Causal language model fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py)
- [Masked language model fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py)
- [Speech pretraining example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py)
- [Translation fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py)
- [Text classification fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py)
- [Semantic segmentation fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/semantic-segmentation/run_semantic_segmentation_no_trainer.py)
- [Question answering fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py)
- [Beam search question answering fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search_no_trainer.py)
- [Multiple choice question answering fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py)
- [Named entity recognition fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py)
- [Image classification fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification_no_trainer.py)
- [Summarization fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py)
- [End-to-end examples on how to use AWS SageMaker integration of Accelerate](https://github.com/huggingface/notebooks/blob/main/sagemaker/22_accelerate_sagemaker_examples/README.md)
- [Megatron-LM examples for various NLp tasks](https://github.com/pacman100/accelerate-megatron-test) 

## 集成示例 

这些是来自与 Accelerate 集成的库的教程： 

> 在这里没有找到您的集成？制作一个 PR 来包含它！

### 安菲翁
- [Training Text-to-Speech Models with Amphion](https://github.com/open-mmlab/Amphion/blob/main/egs/tts/README.md)
- [Training Singing Voice Conversion Models with Amphion](https://github.com/open-mmlab/Amphion/blob/main/egs/svc/README.md)
- [Training Vocoders with Amphion](https://github.com/open-mmlab/Amphion/blob/main/egs/vocoder/README.md)

### 催化剂

- [Distributed training tutorial with Catalyst](https://catalyst-team.github.io/catalyst/tutorials/ddp.html)

### DALLE2-pytorch 

- [Fine-tuning DALLE2](https://github.com/lucidrains/DALLE2-pytorch#usage)

### 扩散器

- [Performing textual inversion with diffusers](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion)
- [Training DreamBooth with diffusers](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)

### 快泰 

- [Distributed training from Jupyter Notebooks with fastai](https://docs.fast.ai/tutorial.distributed.html)
- [Basic distributed training examples with fastai](https://docs.fast.ai/examples/distributed_app_examples.html)

### 毕业生流

- [Auto Image Classification with GradsFlow](https://docs.gradsflow.com/en/latest/examples/nbs/01-ImageClassification/)

### imagen-pytorch 

- [Fine-tuning Imagen](https://github.com/lucidrains/imagen-pytorch#usage)

### 科尼亚

- [Fine-tuning vision models with Kornia's Trainer](https://kornia.readthedocs.io/en/latest/get-started/training.html)

### PyTorch 加速 

- [Quickstart distributed training tutorial with PyTorch Accelerated](https://pytorch-accelerated.readthedocs.io/en/latest/quickstart.html)

### PyTorch3D

- [Perform Deep Learning with 3D data](https://pytorch3d.org/tutorials/)

### 稳定梦想融合- [Training with Stable-Dreamfusion to convert text to a 3D model](https://colab.research.google.com/drive/1MXT3yfOFvO0ooKEfiUUvTKwUkrrlCHpF?usp=sharing)

### 泰兹 

- [Leaf disease detection with Tez and Accelerate](https://www.kaggle.com/code/abhishek/tez-faster-and-easier-training-for-leaf-detection/notebook)

### trlx 

- [How to implement a sentiment learning task with trlx](https://github.com/CarperAI/trlx#example-how-to-add-a-task)

### 舒适的用户界面

- [Enabling using large Stable Diffusion Models in low-vram settings using Accelerate](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py#L291-L296)

## 在科学领域

以下包含使用 Accelerate 的论文的非详尽列表。 

> 在这里找不到您的论文？制作一个 PR 来包含它！

* Yuval Kirstain、Adam Polyak、Uriel Singer、Shahbuland Matiana、Joe Penna、Omer Levy：“Pick-a-Pic：用于文本到图像生成的用户偏好的开放数据集”，2023 年； [arXiv:2305.01569](http://huggingface.co/papers/2305.01569)。
* Lei Wang、Wanyu Xu、Yihuai Lan、Zhiqiang Hu、Yunshi Lan、Roy Ka-Wei Lee、Ee-Peng Lim：“规划与求解提示：通过大型语言模型改进零样本思想链推理”，2023； [arXiv:2305.04091](http://huggingface.co/papers/2305.04091)。
* Arthur Câmara、Claudia Hauff：“移动内容：神经 IR 模型将文档移动到内存的效率研究”，2022 年； [arXiv:2205.08343](http://huggingface.co/papers/2205.08343)。
* Ying Shen、Lianmin Cheng、Binhang Yuan、Zhuohan Li、Max Ryabinin、Daniel Y. Fu、Zhiqiang Xie、Beidi Chen、Clark Ba​​rrett、Joseph E. Gonzalez、Percy Liang、Christopher Ré、Ion Stoica、Ce Zhang：“单 GPU 大型语言模型的高通量生成推理”，2023 年； [arXiv:2303.06865](http://huggingface.co/papers/2303.06865)。
* Peter Melchior、Yan Liang、ChangHoon Hahn、Andy Goulding：“自动编码 Galaxy Spectra I：建筑”，2022 年； [arXiv:2211.07890](http://huggingface.co/papers/2211.07890)。* Jiaao Chen、Aston Zhang、Mu Li、Alex Smola、Diyi Yang：“一种更便宜、更好的带有软掩蔽噪声的扩散语言模型”，2023 年； [arXiv:2304.04746](http://huggingface.co/papers/2304.04746)。
* Ayaan Haque、Matthew Tancik、Aleksander Holynski、Angjoo Kanazawa：“Instruct-NeRF2NeRF：使用指令编辑 3D 场景”，2023 年； [arXiv:2303.12789](http://huggingface.co/papers/2303.12789)。
* Luke Melas-Kyriazi、Christian Rupprecht、Iro Laina、Andrea Vedaldi：“RealFusion：从单个图像 360° 重建任何物体”，2023 年； [arXiv:2302.10663](http://huggingface.co/papers/2302.10663)。
* 吴小石、孙克强、朱峰、赵锐、李宏生：“更好地将文本到图像模型与人类偏好对齐”，2023； [arXiv:2303.14420](http://huggingface.co/papers/2303.14420)。
* 沉永亮、宋凯涛、谭旭、李东升、卢伟明、庄跃亭：“HuggingGPT：利用 ChatGPT 及其 HuggingFace 中的朋友解决人工智能任务”，2023 年； [arXiv:2303.17580](http://huggingface.co/papers/2303.17580)。
* 杨悦、姚文林、张宏明、王晓阳、于东、陈建树：“Z-LaVI：视觉想象力推动的零样本语言求解器”，2022； [arXiv:2210.12261](http://huggingface.co/papers/2210.12261)。
*Sheng-Yen Chou、Pin-Yu Chen、Tsung-Yi Ho：“如何后门扩散模型？”，2022 年； [arXiv:2212.05400](http://huggingface.co/papers/2212.05400)。* Junyoung Seo、Wooseok Jang、Min-Seop Kwak、Jaehoon Ko、Hyeonsu Kim、Junho Kim、Jin-Hwa Kim、Jiyoung Lee、Seungryong Kim：“让 2D 扩散模型了解 3D 一致性，实现稳健的文本到 3D 生成”，2023 年； [arXiv:2303.07937](http://huggingface.co/papers/2303.07937)。
* Or Patashnik、Daniel Garibi、Idan Azuri、Hadar Averbuch-Elor、Daniel Cohen-Or：“使用文本到图像扩散模型本地化对象级形状变化”，2023 年； [arXiv:2303.11306](http://huggingface.co/papers/2303.11306)。
* Dídac Surís、Sachit Menon、Carl Vondrick：“ViperGPT：通过 Python 执行进行视觉推理进行推理”，2023 年； [arXiv:2303.08128](http://huggingface.co/papers/2303.08128)。
* 齐晨阳、村晓东、张勇、雷晨阳、王新涛、单颖、陈奇峰：“FateZero：融合注意力用于零样本文本视频编辑”，2023年； [arXiv:2303.09535](http://huggingface.co/papers/2303.09535)。
* Sean Welleck、Jia Cheng Liu、Ximing Lu、Hannaneh Hajishirzi、Yejin Choi：“NaturalProver：使用语言模型生成扎根数学证明”，2022 年； [arXiv:2205.12910](http://huggingface.co/papers/2205.12910)。
* Elad Richardson、Gal Metzer、Yuval Alaluf、Raja Giryes、Daniel Cohen-Or：“纹理：3D 形状的文本引导纹理”，2023 年； [arXiv:2302.01721](http://huggingface.co/papers/2302.01721)。
* 程普金、林莉、黄一金、何华庆、罗文瀚、唐晓英：“从退化中学习增强：眼底图像增强的扩散模型”，2023； [arXiv:2303.04603](http://huggingface.co/papers/2303.04603)。* Shun Shao、Yftah Ziser、Shay Cohen：“从神经表示中删除未对齐的属性”，2023 年； [arXiv:2302.02997](http://huggingface.co/papers/2302.02997)。
* Seonghyeon Ye、Hyeonbin Hwang、Sohee Yang、Hyungu Yun、Yireun Kim、Minjoon Seo：“情境教学学习”，2023； [arXiv:2302.14691](http://huggingface.co/papers/2302.14691)。
* Shikun Liu、Linxi Fan、Edward Johns、Zhiding Yu、Chaowei Shaw、Anima Anandkumar：“Prismer：专家集合的视觉语言模型”，2023； [arXiv:2303.02506](http://huggingface.co/papers/2303.02506)。
* 陈浩宇、王志华、杨阳、孙奇林、马科德：“学习摄影图像的深色色差度量”，2023 年； [arXiv:2303.14964](http://huggingface.co/papers/2303.14964)。
* Van-Hoang Le，宏宇张：“基于提示的小样本学习的日志解析”，2023； [arXiv:2302.07435](http://huggingface.co/papers/2302.07435)。
* Keito Kudo、Yoichi Aoki、Tatsuki Kuribayashi、Ana Brassard、Masashi Yoshikawa、Keisuke Sakaguchi、Kentaro Inui：“深度神经网络能否捕获算术推理中的组合性？”，2023 年； [arXiv:2302.07866](http://huggingface.co/papers/2302.07866)。
* Ruoyao Wang、Peter Jansen、Marc-Alexandre Côté、Prithviraj Ammanabrolu：“行为克隆变形金刚是神经符号推理者”，2022 年； [arXiv:2210.07382](http://huggingface.co/papers/2210.07382)。* Martin Wessel、Tomáš Horych、Terry Ruas、Akiko Aizawa、Bela Gipp、Timo Spinde：“MBIB 简介——第一个媒体偏见识别基准任务和数据集集合”，2023 年； [arXiv:2304.13148](http://huggingface.co/papers/2304.13148)。 DOI：[https://dx.doi.org/10.1145/3539618.3591882 10.1145/3539618.3591882]。
* Hila Chefer、Yuval Alaluf、Yael Vinker、Lior Wolf、Daniel Cohen-Or：“参与和激发：文本到图像扩散模型的基于注意力的语义指导”，2023 年； [arXiv:2301.13826](http://huggingface.co/papers/2301.13826)。
* Marcio Fonseca、Yftah Ziser、Shay B. Cohen：“在长文档的抽象摘要中分解内容和预算决策”，2022 年； [arXiv:2205.12486](http://huggingface.co/papers/2205.12486)。
* Elad Richardson、Gal Metzer、Yuval Alaluf、Raja Giryes、Daniel Cohen-Or：“纹理：3D 形状的文本引导纹理”，2023 年； [arXiv:2302.01721](http://huggingface.co/papers/2302.01721)。
* 何天行、张靖宇、王天乐、Sachin Kumar、Kyunghyun Cho、James Glass、Yulia Tsvetkov：“论基于模型的文本生成评估指标的盲点”，2022 年； [arXiv:2212.10020](http://huggingface.co/papers/2212.10020)。
* Ori Ram、Yoav Levine、Itay Dalmedigos、Dor Muhlgay、Amnon Shashua、Kevin Leyton-Brown、Yoav Shoham：“上下文检索增强语言模型”，2023 年； [arXiv:2302.00083](http://huggingface.co/papers/2302.00083)。* 李大成、邵儒林、王宏毅、韩国、Eric P. Xing、张浩：“MPCFormer：使用 MPC 进行快速、高性能和私密的 Transformer 推理”，2022 年； [arXiv:2211.01452](http://huggingface.co/papers/2211.01452)。
* Baolin Peng、Michel Galley、Peng Cheng He、Chris Brockett、Lars Liden、Elnaz Nouri、Zhou Yu、Bill Dolan、Jianfeng Taka：“GODEL：目标导向对话的大规模预训练”，2022 年； [arXiv:2206.11309](http://huggingface.co/papers/2206.11309)。
* Egil Rønningstad、Erik Velldal、Lilja Øvrelid：“实体级情感分析 (ELSA)：一项探索性任务调查”，2023 年，第 29 届国际计算语言学会议论文集，2022 年，第 6773-6783 页； [arXiv:2304.14241](http://huggingface.co/papers/2304.14241)。
* Charlie Snell、Ilya Kostrikov、Yi Su、Mengjiao Yang、Sergey Levine：“使用隐式语言 Q 学习进行自然语言生成的离线强化学习”，2022 年； [arXiv:2206.11871](http://huggingface.co/papers/2206.11871)。
*zhiruo Wang、Shuyan Zhou、Daniel Fried、Graham Neubig：“基于执行的开放域代码生成评估”，2022 年； [arXiv:2212.10481](http://huggingface.co/papers/2212.10481)。
* Minh-Long Luu、Zeyi Huang、Eric P. Xing、Yong Jae Lee、Haohan Wang：“通过随机梯度阈值快速显着性引导混合”，2022 年； [arXiv:2212.04875](http://huggingface.co/papers/2212.04875)。
* 刘俊豪、颜汉书、周大全、冯嘉世：“MagicMix：扩散模型的语义混合”，2022； [arXiv:2210.16056](http://huggingface.co/papers/2210.16056)。* Yaqing Wang、Subhabrata Mukherjee、Xiaodong Liu、Jing Gau、Ahmed Hassan Awadallah、Jianfeng Taka：“LiST：Lite Prompted Self-training Makes Parameter-Efficient Few-shot Learners”，2021； [arXiv:2110.06274](http://huggingface.co/papers/2110.06274)。

### 使用 Accelerate 执行梯度累积
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/gradient_accumulation.md