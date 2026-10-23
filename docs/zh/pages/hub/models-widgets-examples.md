<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 小部件示例

请注意，每个小部件示例还可以选择直接在 `output` 属性中描述相应的模型输出。更多详情请参见[the spec](./models-widgets#example-outputs)。

## 自然语言处理

### 填充蒙版

```yaml
widget:
- text: "Paris is the <mask> of France."
  example_title: "Capital"
- text: "The goal of life is <mask>."
  example_title: "Philosophy"
```

### 问答

```yaml
widget:
- text: "What's my name?"
  context: "My name is Clara and I live in Berkeley."
  example_title: "Name"
- text: "Where do I live?"
  context: "My name is Sarah and I live in London"
  example_title: "Location"
```

### 总结

```yaml
widget:
- text: "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
  example_title: "Eiffel Tower"
- text: "Laika, a dog that was the first living creature to be launched into Earth orbit, on board the Soviet artificial satellite Sputnik 2, on November 3, 1957. It was always understood that Laika would not survive the mission, but her actual fate was misrepresented for decades. Laika was a small (13 pounds [6 kg]), even-tempered, mixed-breed dog about two years of age. She was one of a number of stray dogs that were taken into the Soviet spaceflight program after being rescued from the streets. Only female dogs were used because they were considered to be anatomically better suited than males for close confinement."
  example_title: "First in Space"
```

### 表问答

```yaml
widget:
- text: "How many stars does the transformers repository have?"
  table:
    Repository:
      - "Transformers"
      - "Datasets"
      - "Tokenizers"
    Stars:
      - 36542
      - 4512
      - 3934
    Contributors:
      - 651
      - 77
      - 34
    Programming language:
      - "Python"
      - "Python"
      - "Rust, Python and NodeJS"
  example_title: "Github stars"
```

### 文本分类

```yaml
widget:
- text: "I love football so much"
  example_title: "Positive"
- text: "I don't really like this type of food"
  example_title: "Negative"
```

### 文本生成

```yaml
widget:
- text: "My name is Julien and I like to"
  example_title: "Julien"
- text: "My name is Merve and my favorite"
  example_title: "Merve"
```

### Text2Text 生成

```yaml
widget:
- text: "My name is Julien and I like to"
  example_title: "Julien"
- text: "My name is Merve and my favorite"
  example_title: "Merve"
```

### 代币分类

```yaml
widget:
- text: "My name is Sylvain and I live in Paris"
  example_title: "Parisian"
- text: "My name is Sarah and I live in London"
  example_title: "Londoner"
```

### 翻译

```yaml
widget:
- text: "My name is Sylvain and I live in Paris"
  example_title: "Parisian"
- text: "My name is Sarah and I live in London"
  example_title: "Londoner"
```

### 零样本分类

```yaml
widget:
- text: "I have a problem with my car that needs to be resolved asap!!"
  candidate_labels: "urgent, not urgent, phone, tablet, computer"
  multi_class: true
  example_title: "Car problem"
- text: "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app."
  candidate_labels: "mobile, website, billing, account access"
  multi_class: false
  example_title: "Phone issue"
```
### 句子相似度

```yaml
widget:
- source_sentence: "That is a happy person"
  sentences:
    - "That is a happy dog"
    - "That is a very happy person"
    - "Today is a sunny day"
  example_title: "Happy"
```

### 对话

```yaml
widget:
- text: "Hey my name is Julien! How are you?"
  example_title: "Julien"
- text: "Hey my name is Clara! How are you?"
  example_title: "Clara"
```

### 特征提取

```yaml
widget:
- text: "My name is Sylvain and I live in Paris"
  example_title: "Parisian"
- text: "My name is Sarah and I live in London"
  example_title: "Londoner"
```

## 音频

### 文本转语音

```yaml
widget:
- text: "My name is Sylvain and I live in Paris"
  example_title: "Parisian"
- text: "My name is Sarah and I live in London"
  example_title: "Londoner"
```

### 自动语音识别

```yaml
widget:
- src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
  example_title: Librispeech sample 1
- src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
  example_title: Librispeech sample 2
```

### 音频到音频

```yaml
widget:
- src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
  example_title: Librispeech sample 1
- src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
  example_title: Librispeech sample 2
```

### 音频分类

```yaml
widget:
- src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
  example_title: Librispeech sample 1
- src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
  example_title: Librispeech sample 2
```

### 语音活动检测

```yaml
widget:
- src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
  example_title: Librispeech sample 1
- src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
  example_title: Librispeech sample 2
```

## 计算机视觉

### 图像分类

```yaml
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
```

### 物体检测

```yaml
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg
  example_title: Football Match
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg
  example_title: Airport
```

### 图像分割

```yaml
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg
  example_title: Football Match
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg
  example_title: Airport
```

### 图像到图像

```yaml
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/canny-edge.jpg
  prompt: Girl with Pearl Earring # `prompt` field is optional in case the underlying model supports text guidance
```

### 图像到视频

```yaml
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/canny-edge.jpg
  prompt: Girl with Pearl Earring # `prompt` field is optional in case the underlying model supports text guidance
```

### 文本转图像

```yaml
widget:
- text: "A cat playing with a ball"
  example_title: "Cat"
- text: "A dog jumping over a fence"
  example_title: "Dog"
```

### 文档问答

```yaml
widget:
- text: "What is the invoice number?"
  src: "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"
- text: "What is the purchase amount?"
  src: "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/contract.jpeg"
```

### 视觉问答

```yaml
widget:
- text: "What animal is it?"
  src: "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"
- text: "Where is it?"
  src: "https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg"
```

### 零样本图像分类

```yaml
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/cat-dog-music.png
  candidate_labels: playing music, playing sports
  example_title: Cat & Dog
```

## 其他

### 结构化数据分类

```yaml
widget:
- structured_data:
    fixed_acidity:
      - 7.4
      - 7.8
      - 10.3
    volatile_acidity:
      - 0.7
      - 0.88
      - 0.32
    citric_acid:
      - 0
      - 0
      - 0.45
    residual_sugar:
      - 1.9
      - 2.6
      - 6.4
    chlorides:
      - 0.076
      - 0.098
      - 0.073
    free_sulfur_dioxide:
      - 11
      - 25
      - 5
    total_sulfur_dioxide:
      - 34
      - 67
      - 13
    density:
      - 0.9978
      - 0.9968
      - 0.9976
    pH:
      - 3.51
      - 3.2
      - 3.23
    sulphates:
      - 0.56
      - 0.68
      - 0.82
    alcohol:
      - 9.4
      - 9.8
      - 12.6
  example_title: "Wine"
```### Spaces ZeroGPU：空间的动态 GPU 分配
https://huggingface.co/docs/hub/spaces-zerogpu.md