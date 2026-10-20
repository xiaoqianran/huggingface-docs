# Talks

This page lists talks on the Kernels project, delivered at
different events:

* [Lecture 106: Hugging Face Kernels (GPU Mode)](https://www.youtube.com/watch?v=Ok8vi6JemVQ)
* [PyTorch Day India 2026 Beyond the Brrr: Building a Unified Ecosystem for Optimized Kernels](https://www.youtube.com/watch?v=q0GfzJmuaUM)
* [Build a PyTorch ReLU Kernel with Hugging Face Kernels (CPU + Metal)](https://youtu.be/wQR-QC7pbqQ?is=JY4pvjXpkz8ghgwS)

This page will be kept updated.

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

### kernels info
https://huggingface.co/docs/kernels/main/cli-info.md
