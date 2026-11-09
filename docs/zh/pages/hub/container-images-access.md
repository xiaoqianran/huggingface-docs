<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 容器镜像访问控制

谁可以拉取和推送镜像取决于存储库的可见性以及每个成员在组织中的角色：任何人都可以拉取公共镜像，只有成员可以拉取私有镜像，并且推送需要`write`角色。代币遵循相同的规则：一个带有 `read` 拉动的 [User Access Token](./security-tokens)，一个带有 `write` 推动的 [User Access Token](./security-tokens)。

## 资源组的访问控制

在采用团队或企业计划的组织中，映像可以属于 [Resource Group](./security-resource-groups)。然后，组中成员的角色将替换该映像的组织角色：`read` 进行拉取，`write` 进行推送。资源组中的私有映像只能由该组的成员和组织管理员拉取。

创建存储库时，从图像的设置或组的 **资源** 选项卡中选择组。使用 API，在创建存储库时传递 `resourceGroupId`，或移动现有存储库（组 ID 以其页面 URL 结尾，`GET /api/organizations/my-org/resource-groups` 列出它们）：

```bash
curl -X POST https://huggingface.co/api/containers/my-org/my-app/resource-group \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"resourceGroupId": "<resource-group-id>"}'
```

### 发布商分析
https://huggingface.co/docs/hub/publisher-analytics.md