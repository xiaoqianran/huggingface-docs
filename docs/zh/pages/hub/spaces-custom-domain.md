<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间自定义域

> [!警告]
> 此功能是 PRO 或团队和企业计划的一部分。

## 自定义域入门

空间自定义域允许您在您选择的自定义域上托管您的空间：`yourdomain.example.com`。自定义域必须是有效的 DNS 名称。

> [!注意]
> 自定义域要求您的空间具有 **公共** 或 **受保护** 可见性。私人空间不支持它们。

    
    

### 设置您的域

您可以在空间设置中的“自定义域”下提交自定义域。您需要添加一条 CNAME 记录，将您的域指向 `hf.space`：

    
    

### 验证您的域名

提交后，请求将转为“待处理”状态：

    
    

正确配置 DNS 后，您将看到“就绪”状态，确认自定义域在您的空间中处于活动状态。

如果您已完成所有步骤，但没有看到“就绪”状态，则可以输入您的域 [here](https://toolbox.googleapps.com/apps/dig/#CNAME/) 来验证它是否指向 `hf.space`。如果没有，请检查您的域名托管服务商以确保 CNAME 记录已正确添加。

## 删除自定义域只需使用空间设置中“自定义域”右侧的删除按钮即可删除自定义域。您可以在自定义域处于待处理或就绪状态时删除。

### 组织中的访问控制
https://huggingface.co/docs/hub/organizations-security.md