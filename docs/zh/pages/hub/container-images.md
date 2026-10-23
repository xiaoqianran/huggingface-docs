<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 容器镜像

容器镜像是 Hugging Face Hub 上用于 Docker 和其他 [OCI](https://opencontainers.org/) 镜像的存储库类型。 Hugging Face 容器注册表 `cr.hf.co` 将图像存储在 [Xet](./xet/index) 存储后端。使用推荐客户端[⟦T5⟧](#pushing-and-pulling-with-hf-image)推拉它们； Docker、Podman、crane、skopeo、ORAS 和 Helm 等标准客户端也可以工作。

> [!注意]
> 容器镜像处于私人测试版：[contact us](https://huggingface.co/contact/sales)，以便为您的组织启用它们。

## 创建存储库

转到[huggingface.co/new-container](https://huggingface.co/new-container)并选择组织、名称和可见性。或者使用API：

```bash
curl -X POST https://huggingface.co/api/containers/my-org/my-app \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"private": true}'
```

在推送之前创建存储库：推送到缺失的存储库会失败，并显示 `NAME_UNKNOWN`。它的页面`https://huggingface.co/containers/my-org/my-app`列出了它的标签。谁可以拉动和推送取决于存储库的可见性以及每个成员在组织中的角色。

## 使用 `hf image` 推拉

[⟦T9⟧](https://github.com/huggingface/hf-image)是`hf` CLI的扩展，仅传输自上一版本以来各层发生变化的部分，与其他OCI客户端相比，体验更快。

```bash
hf extensions install hf-image
hf auth login                                   # or set HF_TOKEN
hf image build -t cr.hf.co/my-org/my-app:v2 .   # optional: docker build works too
hf image push cr.hf.co/my-org/my-app:v2
hf image pull cr.hf.co/my-org/my-app:v2
```

它使用你的 Hugging Face [User Access Token](./security-tokens)（`read` 来拉，`write` 来推），并且需要 Docker 和 [containerd image store](https://docs.docker.com/engine/storage/containerd/)。使用 `hf image` 推送的图像可以由任何客户端拉取。由于 `hf image` 存储未压缩的图层，因此它推送的图像会获得与 `docker push` 不同的摘要。要保留 `docker push` 摘要，对于摘要固定的签名或部署，请使用 `hf image push --preserve-digests`。它比较慢：更改后的图层会被完整地推送和拉动，就像`docker push`一样。

## 使用 Docker 和其他 OCI 客户端

虽然我们建议使用 `hf image`（请参阅上一节）来充分利用 [Xet backend](./xet/index)，但您也可以通过任何 OCI 客户端与注册表进行交互。

使用您的 Hugging Face 用户名和 [User Access Token](./security-tokens) 作为密码登录：`read` 用于拉取，`write` 用于推送。无需登录即可拉取公共镜像。

```bash
echo $HF_TOKEN | docker login cr.hf.co -u <username> --password-stdin
docker tag my-app cr.hf.co/my-org/my-app:v1
docker push cr.hf.co/my-org/my-app:v1
docker pull cr.hf.co/my-org/my-app:v1
```

其他 OCI 客户端的工作方式相同。 Helm 将图表名称附加到 URL：`helm push mychart-0.1.0.tgz oci://cr.hf.co/my-org` 推送到 `my-org/mychart`，它必须事先存在。

## 删除图像

使用[⟦T24⟧](#pushing-and-pulling-with-hf-image)，删除图像的标签，然后通过摘要删除它：

```bash
hf image untag cr.hf.co/my-org/my-app:v1                # prints the image's digest
hf image delete cr.hf.co/my-org/my-app@sha256:<digest>
```

如果某些内容仍然引用该图像，例如另一个标签，则删除将被拒绝并列出首先要删除的内容。没有其他图像使用的图层稍后会被删除。

## 限制

- 图像名称为 `cr.hf.co/<namespace>/<name>`，小写。
- 层高达 100 GB，显示高达 4 MiB。
- 标签列表可能需要几秒钟的时间才能显示新标签。### 在拥抱脸部时使用 Stanza
https://huggingface.co/docs/hub/stanza.md