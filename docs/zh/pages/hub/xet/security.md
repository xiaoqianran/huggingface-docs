<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 安全模型

Xet 存储对 Hugging Face 中存储的所有块提供重复数据删除。这是通过加密哈希以隐私敏感的方式完成的。块的内容受到保护并与存储库权限相关联，即您只能读取重现您有权访问的文件所需的块，而不能读取更多块。

有关如何以保护隐私的方式完成重复数据删除的更多信息和细节，请参阅[Xet Protocol Specification](https://huggingface.co/docs/xet/deduplication)。

### 重复数据删除
https://huggingface.co/docs/hub/xet/deduplication.md