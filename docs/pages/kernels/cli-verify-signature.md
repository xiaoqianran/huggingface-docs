# kernels verify-signature

Use `kernels verify-signature` to verify the metadata signature and check
that kernel files match the digest embedded in the metadata.

## Usage

```bash
kernels verify-signature <repo_id> <version> [--all-variants] \
  [--filter-unsigned] [--filter-no-digest]
```

## What It Does

- Checks that the signing identity in `metadata.json.sigstore` is approved.
- Verifies that `metadata.json` is not tampered with, using the signature
  in `metadata.json.sigstore`.
- Verifies that other kernel files are not tampered with, using the digest
  in `metadata.json`.

## Examples

Verify version `1` of the `kernels-community/relu` kernel. Only checks
the variant that is compatible with the current system:

```bash
kernels verify-signature kernels-community/relu 1
```

Verify all build variants of the same kernel:

```bash
kernels verify-signature kernels-community/relu 1 --all-variants
```

## Example Output

```bash
$ kernels verify-signature kernels-community/relu 1
✅ torch211-cxx11-cu126-x86_64-linux: kernel metadata is correctly signed
$ kernels verify-signature kernels-community/flash-attn2 1
❌ torch211-cxx11-cu126-x86_64-linux: cannot verify kernel integrity, signature not found
```

## Options

| Option               | Description                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--all-variants`     | Verify all build variants of each kernel instead of just the variant that is compatible with the current system. |
| `--filter-no-digest` | Skip variants that do not have a digest in the metadata (typically older builds that precede code signing).      |
| `--filter-unsigned`  | Skip variants that do not have a detached signature (typically older builds that precede code signing).          |

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

### Migrate from older versions
https://huggingface.co/docs/kernels/main/migration.md
