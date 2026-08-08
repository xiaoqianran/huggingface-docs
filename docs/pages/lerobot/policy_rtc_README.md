# Real-Time Chunking (RTC)

This module contains the LeRobot implementation of **Real-Time Chunking (RTC)**, an inference-time technique for flow-matching based policies.

**Note**: RTC is not a policy itself, but rather an inference enhancement that works with flow-matching based policies including [π₀](../pi0/), [π₀.₅](../pi05/), and [SmolVLA](../smolvla/).

---

## Citation

If you use Real-Time Chunking in your work, please cite:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{black2025realtimeexecutionactionchunking,
      title={Real-Time Execution of Action Chunking Flow Policies},
      author={Kevin Black and Manuel Y. Galliker and Sergey Levine},
      year={2025},
      eprint={2506.07339},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.07339},
}
```

---

## License

This implementation follows the **Apache 2.0 License**, consistent with the LeRobot project.

### Paper
https://huggingface.co/docs/lerobot/policy_smolvla_README.md

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

### Implement your own Robot Processor
https://huggingface.co/docs/lerobot/implement_your_own_processor.md
