# Installation

🤗 Tokenizers is tested on Python 3.5+.

You should install 🤗 Tokenizers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're
unfamiliar with Python virtual environments, check out the [user
guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
Create a virtual environment with the version of Python you're going to
use and activate it.

## Installation with pip

🤗 Tokenizers can be installed using pip as follows:

```bash
pip install tokenizers
```

## Installation from sources

To use this method, you need to have the Rust language installed. You
can follow [the official
guide](https://www.rust-lang.org/learn/get-started) for more
information.

If you are using a unix based OS, the installation should be as simple
as running:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Or you can easily update it with the following command:

```bash
rustup update
```

Once rust is installed, we can start retrieving the sources for 🤗
Tokenizers:

```bash
git clone https://github.com/huggingface/tokenizers
```

Then we go into the python bindings folder:

```bash
cd tokenizers/bindings/python
```

At this point you should have your [virtual environment]() already
activated. In order to compile 🤗 Tokenizers, you need to:

```bash
pip install -e .
```

## Crates.io

🤗 Tokenizers is available on [crates.io](https://crates.io/crates/tokenizers).

You just need to add it to your `Cargo.toml`:

```bash
cargo add tokenizers
```

## Installation with npm

You can simply install 🤗 Tokenizers with npm using:

```bash
npm install tokenizers
```

### Training from memory
https://huggingface.co/docs/tokenizers/v0.23.2/training_from_memory.md
