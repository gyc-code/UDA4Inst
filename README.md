# UDA4Inst: Unsupervised Domain Adaptation for Instance Segmentation

[[`arXiv`](https://arxiv.org/abs/2405.09682)] [[`Project`](https://github.com/gyc-code/UDA4Inst)] [[`BibTeX`](...)]

# Architecture
[Fig2.pdf](https://github.com/user-attachments/files/19905592/Fig2.pdf)

[Fig3.pdf](https://github.com/user-attachments/files/19905596/Fig3.pdf)



### Features

* A UDA architecture for instance segmentation from synthetic domain to real domain.
* Support synthetic and real domain segmentation datasets: Urbansyn, Synscapes, SYNTHIA, Cityscapes, KITTI360.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started
### Train
bash train_net.sh


## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Mask2Former Model Zoo](MODEL_ZOO.md).

## License

Code is largely based on Mask2Former.
Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
The majority of Mask2Former is licensed under a [MIT License](LICENSE).

## <a name="CitingMask2Former"></a>Citing Mask2Former

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@article{guo2024uda4inst,
  title={UDA4Inst: Unsupervised Domain Adaptation for Instance Segmentation},
  author={Guo, Yachan and Xiao, Yi and Xue, Danna and Zurita, Jose Luis Gomez and L{\'o}pez, Antonio M},
  journal={arXiv preprint arXiv:2405.09682},
  year={2024}
}
```

## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).
