# Using PaddleNLP at Hugging Face

Leveraging the [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) framework, [`PaddleNLP`](https://github.com/PaddlePaddle/PaddleNLP) is an easy-to-use and powerful NLP library with awesome pre-trained model zoo, supporting wide-range of NLP tasks from research to industrial applications.

## Exploring PaddleNLP in the Hub

You can find `PaddleNLP` models by filtering at the left of the [models page](https://huggingface.co/models?library=paddlenlp&sort=downloads).

All models on the Hub come up with the following features:

1. An automatically generated model card with a brief description and metadata tags that help for discoverability.
2. An interactive widget you can use to play out with the model directly in the browser.

3. An Inference Providers widget that allows to make inference requests.

4. Easily deploy your model as a Gradio app on Spaces.

## Installation

To get started, you can follow [PaddlePaddle Quick Start](https://www.paddlepaddle.org.cn/en/install) to install the PaddlePaddle Framework with your favorite OS, Package Manager and Compute Platform.

`paddlenlp` offers a quick one-line install through pip:

```
pip install -U paddlenlp
```

## Using existing models

Similar to `transformer` models, the `paddlenlp` library provides a simple one-liner to load models from the Hugging Face Hub by setting `from_hf_hub=True`! Depending on how you want to use them, you can use the high-level API using the `Taskflow` function or you can use `AutoModel` and `AutoTokenizer` for more control.

```py
# Taskflow provides a simple end-to-end capability and a more optimized experience for inference
from paddlenlp.transformers import Taskflow
taskflow = Taskflow("fill-mask", task_path="PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)

# If you want more control, you will need to define the tokenizer and model.
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)
model = AutoModelForMaskedLM.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)
```

If you want to see how to load a specific model, you can click `Use in paddlenlp` and you will be given a working snippet that you can load it!

## Sharing your models

You can share your `PaddleNLP` models by using the `save_to_hf_hub` method under all `Model` and `Tokenizer` classes.

```py
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)
model = AutoModelForMaskedLM.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)

tokenizer.save_to_hf_hub(repo_id="<my_org_name>/<my_repo_name>")
model.save_to_hf_hub(repo_id="<my_org_name>/<my_repo_name>")
```

## Additional resources

- PaddlePaddle Installation [guide](https://www.paddlepaddle.org.cn/en/install).
- PaddleNLP [GitHub Repo](https://github.com/PaddlePaddle/PaddleNLP).
- [PaddlePaddle on the Hugging Face Hub](https://huggingface.co/PaddlePaddle)

### TF-Keras (legacy)
https://huggingface.co/docs/hub/tf-keras.md

## TF-Keras (legacy)

`tf-keras` is the name given to Keras 2.x version. It is now hosted as a separate GitHub repo [here](https://github.com/keras-team/tf-keras). Though it's a legacy framework, there are still [4.5k+ models](https://huggingface.co/models?library=tf-keras&sort=trending) hosted on the Hub. These models can be loaded using the `huggingface_hub` library. You **must** have either `tf-keras` or `keras<3.x` installed on your machine.

If you are interested in Keras 3.x support, check out [this guide](./keras).

Once installed, you just need to use the `from_pretrained_keras` method to load a model from the Hub. Read more about `from_pretrained_keras` [here](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_keras).

```py
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("keras-io/mobile-vit-xxs")
prediction = model.predict(image)
prediction = tf.squeeze(tf.round(prediction))
print(f'The image is a {classes[(np.argmax(prediction))]}!')

<CopyLLMTxtMenu containerStyle="float: right; margin-left: 10px; display: inline-flex; position: relative; z-index: 10;"></CopyLLMTxtMenu>

# The image is a sunflower!
```

You can also host your `tf-keras` model on the Hub. However, keep in mind that `tf-keras` is a legacy framework. To reach a maximum number of users, we recommend to create your model using Keras 3.x and share it natively as described above. For more details about uploading `tf-keras` models, check out [`push_to_hub_keras` documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.push_to_hub_keras).

```py
from huggingface_hub import push_to_hub_keras

push_to_hub_keras(model,
    "your-username/your-model-name",
    "your-tensorboard-log-directory",
    tags = ["object-detection", "some_other_tag"],
    **model_save_kwargs,
)
```

## Additional resources

- [GitHub repo](https://github.com/keras-team/tf-keras)
* Blog post [Putting Keras on 🤗 Hub for Collaborative Training and Reproducibility](https://merveenoyan.medium.com/putting-keras-on-hub-for-collaborative-training-and-reproducibility-9018301de877) (April 2022)

### Argilla on Spaces
https://huggingface.co/docs/hub/spaces-sdks-docker-argilla.md
