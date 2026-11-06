# 🤗 LeRobot Notebooks

This repository contains example notebooks for using LeRobot. These notebooks demonstrate how to train policies on real or simulation datasets using standardized policies.

---

### Training ACT

[ACT](https://huggingface.co/papers/2304.13705) (Action Chunking Transformer) is a transformer-based policy architecture for imitation learning that processes robot states and camera inputs to generate smooth, chunked action sequences.

We provide a ready-to-run Google Colab notebook to help you train ACT policies using datasets from the Hugging Face Hub.

| Notebook                                                                                                | Colab                                                                                                                                                                             |
| :------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Train ACT with LeRobot](https://github.com/huggingface/notebooks/blob/main/lerobot/training-act.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb) |

Expected training time for 100k steps: ~1.5 hours on an NVIDIA A100 GPU with batch size of `64`.

### Training SmolVLA

[SmolVLA](https://huggingface.co/papers/2506.01844) is a small but efficient Vision-Language-Action model. It is compact in size with 450 M-parameter and is developed by Hugging Face.

We provide a ready-to-run Google Colab notebook to help you train SmolVLA policies using datasets from the Hugging Face Hub.

| Notebook                                                                                                        | Colab                                                                                                                                                                                 |
| :-------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Train SmolVLA with LeRobot](https://github.com/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb) |

Expected training time for 20k steps: ~5 hours on an NVIDIA A100 GPU with batch size of `64`.

### Paper
https://huggingface.co/docs/lerobot/v0.7.0/policy_smolvla_README.md

## Paper

https://arxiv.org/abs/2506.01844

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```

### Damiao Motors and CAN Bus
https://huggingface.co/docs/lerobot/v0.7.0/damiao.md
