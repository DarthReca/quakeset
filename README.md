<h1 align="center">
  QuakeSet
</h1>

<div align="center">
<br />

[![Project license](https://img.shields.io/github/license/DarthReca/quakeset.svg?style=flat-square)](LICENSE)

[![code with love by DarthReca](https://img.shields.io/badge/%3C%2F%3E%20with%20%E2%99%A5%20by-DarthReca-ff1414.svg?style=flat-square)](https://github.com/DarthReca)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Authors & contributors](#authors--contributors)
- [License](#license)

</details>

---

## About

This is the repository to reproduce the experiments presented in _QuakeSet: A Dataset and
Low-Resource Models to Monitor Earthquakes through Sentinel-1_ accepted at ISCRAM 2024.

The dataset is available on [HuggingFace](https://huggingface.co/datasets/DarthReca/quakeset) or from [TorchGeo](https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/quakeset.py).

PrePrint: https://arxiv.org/abs/2403.18116

**REPOSITORY IN CONSTRUCTION: SOME FILES COULD BE MISSING**

## Getting Started

### Prerequisites

The project is built upon:
- SkLearn
- PyTorch Lightning
- Hydra
- Timm
- Transformers

## Usage

The experiments for classical machine learning models can be reproduced with *classical_main.py*. The experiments with deep learning models can be done with *main.py*.

You can change the parameters in the *configs* folder or through the command line. 

## Authors & contributors

The original setup of this repository is by [Daniele Rege Cambrin](https://github.com/DarthReca).

For a full list of all authors and contributors, see [the contributors page](https://github.com/DarthReca/quakeset/contributors).

## License

This project is licensed under the **Apache 2.0 license**.

See [LICENSE](LICENSE) for more information.
