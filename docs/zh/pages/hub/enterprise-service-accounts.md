<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 服务帐户

> [!警告]
> 此功能是 Enterprise 和 Enterprise Plus 计划的一部分。

服务帐户是组织拥有的身份，旨在以编程方式访问组织的资源，例如 CI/CD 管道、自动化脚本和后端集成。与个人会员帐户不同，服务帐户不与个人绑定：它属于组织并由其管理员管理。

由于服务帐户与任何个人分离，因此即使人们加入或离开组织，它也能保持自动化工作流程运行，并且它允许您在不影响任何成员的个人令牌的情况下确定范围和轮换凭证。

## 创建服务帐户

作为组织管理员，请转到组织设置的 **服务帐户** 部分来创建和管理服务帐户。

  
  

创建服务帐户时，您需要提供：

- 用于标识服务帐户的**名称**。
- 可选的**描述**，用于记录服务帐户的用途。服务帐户没有密码，无法交互式登录 - 它们只能通过您为其颁发的访问令牌进行访问。与普通用户不同，他们也没有 HF 个人资料页面，不属于组织的成员列表，并且只能通过 **服务帐户** 设置进行管理。

## 管理访问令牌

服务帐户对您的组织的访问完全由您向其颁发的细粒度访问令牌定义。对于每个令牌，您可以选择一个名称和一组细粒度的权限，以便您可以仅授予给定工作流所需的访问权限。

可以在两个级别授予权限：

- **组织范围** — 在组织中的所有存储库中应用权限（例如，对存储库内容的读取或写入访问权限）。
- **每个存储库** — 仅对特定存储库进行读或写访问。搜索并选择令牌应应用到的存储库，然后选择要授予的权限。选定的存储库必须归组织所有或公开。这使您可以发出范围狭窄的令牌（例如，只能读取单个模型存储库的令牌），而不是授予对整个组织的访问权限。

  
  

从服务帐户的页面，管理员可以：

- **创建**具有所选名称和细粒度权限的新访问令牌。
- **更新**现有令牌的名称或权限。
- **旋转**令牌以将其替换为新值。之前的令牌会立即停止工作，如果凭证可能已泄露，这会很有用。
- **删除**令牌以永久撤销其访问权限。

> [!警告]
> 访问令牌仅在创建或轮换时显示一次。安全地保存它——以后无法取回。如果丢失，请旋转令牌以生成新值。

### 网络安全豁免

> [!警告]
> 此选项是 Enterprise Plus 计划的一部分。

如果您的组织使用 [Network Security](./enterprise-network-security) 设置，则服务帐户令牌将像任何其他凭据一样受到它们的约束：它们仅在您组织的 IP 范围内工作，并且内容访问策略适用于它们可以访问的范围。一些自动化工作流程在公司网络外部运行 - 例如，由云提供商托管的 CI 作业。对于这些情况，您可以使用令牌表单中的 **绕过 IP 限制和内容访问策略** 选项，使单个令牌免于组织的网络安全强制执行。

启用后：

- 令牌可在任何 IP 地址上使用，即使**仅限制组织访问您的 IP 范围**打开也是如此。
- 令牌可以访问您组织的内容访问策略否则会阻止的内容。
- 豁免仅适用于您自己组织的网络安全设置。它永远不会绕过其他组织的策略。
- 令牌在服务帐户页面上列出，并带有 **无 IP 限制** 徽章。
- 轮换令牌可保留豁免。

只有组织管理员才能启用豁免或轮换已豁免的令牌。由于豁免令牌不再受到网络边界的保护，因此令牌本身成为保护其可以访问的资源的唯一凭证。我们建议将其范围限制为尽可能小的存储库集，在足够的情况下授予只读权限，并定期轮换它。

## 计费

服务帐户不计为组织的计费成员，因此创建它们不会占用计划中的付费席位。

### TF-Keras（旧版）
https://huggingface.co/docs/hub/tf-keras.md

## TF-Keras（旧版）

`tf-keras` 是 Keras 2.x 版本的名称。它现在作为单独的 GitHub 存储库 [here](https://github.com/keras-team/tf-keras) 托管。尽管它是一个遗留框架，但 Hub 上仍然托管着 [4.5k+ models](https://huggingface.co/models?library=tf-keras&sort=trending)。这些模型可以使用 `huggingface_hub` 库加载。您**必须**在您的计算机上安装 `tf-keras` 或 `keras<3.x`。

如果您对 Keras 3.x 支持感兴趣，请查看 [this guide](./keras)。

安装后，您只需使用`from_pretrained_keras`方法从Hub加载模型即可。了解更多关于 `from_pretrained_keras` [here](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_keras) 的信息。

```py
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("keras-io/mobile-vit-xxs")
prediction = model.predict(image)
prediction = tf.squeeze(tf.round(prediction))
print(f'The image is a {classes[(np.argmax(prediction))]}!')

<CopyLLMTxtMenu containerStyle="float: right; margin-left: 10px; display: inline-flex; position: relative; z-index: 10;"></CopyLLMTxtMenu>

# The image is a sunflower!
```您还可以在 Hub 上托管您的 `tf-keras` 模型。但是，请记住，`tf-keras` 是一个遗留框架。为了达到最大数量的用户，我们建议使用 Keras 3.x 创建模型并如上所述在本机共享。有关上传`tf-keras`模型的更多详细信息，请查看[⟦T11⟧ documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.push_to_hub_keras)。

```py
from huggingface_hub import push_to_hub_keras

push_to_hub_keras(model,
    "your-username/your-model-name",
    "your-tensorboard-log-directory",
    tags = ["object-detection", "some_other_tag"],
    **model_save_kwargs,
)
```

## 其他资源

- [GitHub repo](https://github.com/keras-team/tf-keras)
* 博文[Putting Keras on 🤗 Hub for Collaborative Training and Reproducibility](https://merveenoyan.medium.com/putting-keras-on-hub-for-collaborative-training-and-reproducibility-9018301de877)（2022 年 4 月）

### Hugging Face Hub 上的 Jupyter 笔记本
https://huggingface.co/docs/hub/notebooks.md