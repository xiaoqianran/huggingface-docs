<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 集线器上的存储区域

> [!警告]
> 此功能是团队和企业计划的一部分。

区域允许您指定组织的模型、数据集、空间和存储桶的存储位置。对于非团队或企业用户，存储库始终存储在美国。

这提供了两个主要好处：

- 监管和法律合规性
- 性能（更快的下载/上传速度和更低的延迟）

目前可用地区：

- 美国🇺🇸
- 欧盟🇪🇺
- 即将推出：亚太地区 🌏
- 即将推出：海湾合作委员会（沙特阿拉伯）🇸🇦

## 存储区域入门

订阅团队或企业计划的组织可以访问区域设置页面来管理其存储库存储位置。

  <img
    class="block dark:hidden m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/regions.png"
    alt="screenshot of Hugging Face Storage Regions feature"
  />
  <img
    class="hidden dark:block m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-regions.png"
    alt="screenshot of Hugging Face Storage Regions feature"
  />

该页面显示：

- 对您组织的存储库位置的审核
- 选择新存储库存储位置的选项

> [!提示]
> Spaces 的一些高级计算选项（例如 ZeroGPU）可能并非在所有区域都可用。

## 存储库标签

存储在非默认位置的任何存储库（模型或数据集）都会将其区域显示为标签，从而允许组织成员快速识别存储库位置。

  <img
    class="block dark:hidden m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/region-tag.png"
    alt="screenshot of Hugging Face Storage Regions tag feature"
  />
  <img
    class="hidden dark:block m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-region-tag.png"
    alt="screenshot of Hugging Face Storage Regions tag feature"
  />

## 监管和法律合规性受监管的行业通常需要在特定区域存储数据。

对于欧盟公司，您可以以符合 GDPR 的方式使用 Hub 进行机器学习开发，数据集、模型、存储桶和推理端点均存储在欧盟数据中心。

## 性能

将模型和数据集存储在离您的团队和基础设施更近的地方可以显着提高上传和下载的性能。

鉴于模型权重和数据集文件通常较大，这种影响是巨大的。

  <img
    class="block dark:hidden m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/region-git-code.png"
    alt="example of Hugging Face Storage Regions feature"
  />
  <img
    class="hidden dark:block m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-region-git-code.png"
    alt="example of Hugging Face Storage Regions feature"
  />

例如，与美国存储相比，在欧盟地区存储存储库的欧洲用户预计上传和下载速度大约快 4-5 倍。

## 空格

两个空间的存储和运行时都使用所选区域。

可用的硬件配置因地区而异，并且某些功能可能并非在所有地区都可用。如需特殊要求，请联系您的 HF 客户团队。

### 如何向 ArXiv 添加空间
https://huggingface.co/docs/hub/spaces-add-to-arxiv.md