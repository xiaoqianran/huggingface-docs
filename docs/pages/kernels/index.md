# Kernels

## Docs

- [Kernels](https://huggingface.co/docs/kernels/main/index.md)
- [Use layers](https://huggingface.co/docs/kernels/main/layers.md)
- [kernels download](https://huggingface.co/docs/kernels/main/cli-download.md)
- [kernels benchmark](https://huggingface.co/docs/kernels/main/cli-benchmark.md)
- [kernels versions](https://huggingface.co/docs/kernels/main/cli-versions.md)
- [kernels verify-signature](https://huggingface.co/docs/kernels/main/cli-verify-signature.md)
- [Why kernels?](https://huggingface.co/docs/kernels/main/why_kernels.md)
- [Overview](https://huggingface.co/docs/kernels/main/cli.md)
- [kernels variants](https://huggingface.co/docs/kernels/main/cli-variants.md)
- [Migrate from older versions](https://huggingface.co/docs/kernels/main/migration.md)
- [Installation](https://huggingface.co/docs/kernels/main/installation.md)
- [FAQ](https://huggingface.co/docs/kernels/main/faq.md)
- [Talks](https://huggingface.co/docs/kernels/main/talks.md)
- [kernels lock](https://huggingface.co/docs/kernels/main/cli-lock.md)
- [Quickstart](https://huggingface.co/docs/kernels/main/basic-usage.md)
- [kernels info](https://huggingface.co/docs/kernels/main/cli-info.md)
- [CLI reference for kernel-builder](https://huggingface.co/docs/kernels/main/builder-cli.md)
- [Kernel requirements](https://huggingface.co/docs/kernels/main/kernel-requirements.md)
- [Blog posts](https://huggingface.co/docs/kernels/main/blog.md)
- [kernels cache ls](https://huggingface.co/docs/kernels/main/cli-cache.md)
- [kernels describe-api](https://huggingface.co/docs/kernels/main/cli-describe-api.md)
- [Install agent skills](https://huggingface.co/docs/kernels/main/cli-skills.md)
- [Environment variables](https://huggingface.co/docs/kernels/main/env.md)
- [Kernels security](https://huggingface.co/docs/kernels/main/security.md)
- [Lock kernel versions](https://huggingface.co/docs/kernels/main/locking.md)
- [Projects using kernels](https://huggingface.co/docs/kernels/main/integrating-kernels.md)
- [Layers API Reference](https://huggingface.co/docs/kernels/main/api/layers.md)
- [Kernels API Reference](https://huggingface.co/docs/kernels/main/api/kernels.md)
- [Set up your IDE](https://huggingface.co/docs/kernels/main/builder/ide-setup.md)
- [Set up for Metal kernels](https://huggingface.co/docs/kernels/main/builder/metal.md)
- [Develop locally](https://huggingface.co/docs/kernels/main/builder/local-dev.md)
- [Build with Nix](https://huggingface.co/docs/kernels/main/builder/build.md)
- [Build variants](https://huggingface.co/docs/kernels/main/builder/build-variants.md)
- [Why Nix?](https://huggingface.co/docs/kernels/main/builder/why-nix.md)
- [Develop kernels with agents](https://huggingface.co/docs/kernels/main/builder/agents-guide.md)
- [Architecture overview](https://huggingface.co/docs/kernels/main/builder/design-overview.md)
- [Ship Triton autotune configurations](https://huggingface.co/docs/kernels/main/builder/triton-autotune.md)
- [Nix builder design](https://huggingface.co/docs/kernels/main/builder/design-nix-builder.md)
- [Write kernels](https://huggingface.co/docs/kernels/main/builder/writing-kernels.md)
- [Building and testing kernels with GitHub Actions](https://huggingface.co/docs/kernels/main/builder/github-actions.md)
- [Secure your kernels](https://huggingface.co/docs/kernels/main/builder/security.md)

### Kernels
https://huggingface.co/docs/kernels/main/index.md

# Kernels

The Kernel Hub allows Python libraries and applications to load compute
kernels directly from the [Hub](https://huggingface.co/). Kernels are a first-class
repository type on the Hub, with dedicated pages that surface supported
hardware and versions. To support dynamic loading, Hub kernels differ from
traditional Python kernel packages in that they are made to be:

- **Portable**: a kernel can be loaded from paths outside `PYTHONPATH`.
- **Unique**: multiple versions of the same kernel can be loaded in the
  same Python process.
- **Compatible**: `kernels` must support all recent versions of Python and
  the different PyTorch build configurations (various CUDA versions
  and C++ ABIs). Furthermore, older C library versions must be supported.

Browse available kernels at [huggingface.co/kernels](https://huggingface.co/kernels).

The Kernels project is divided into two parts:

- Builder: [`kernel-builder`](builder-cli) provides utilities to build, package, and distribute compute kernels in a way that is compatible with the Hugging Face Hub and `kernels`.
- `kernels`: The [`kernels`](basic-usage) is a Python package that lets
  users load compatible compute kernels from the Hub. Refer to the [quickstart](basic-usage) to know more.

If you're looking for a more involved "Why kernels?" answer, refer to
[this page](./why_kernels).

The [talks page](./talks) page has links to talks on the
Kernels project. The [blog page](./blog) collects blog posts
on the Kernels project.

### Use layers
https://huggingface.co/docs/kernels/main/layers.md
