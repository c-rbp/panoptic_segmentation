Quick build detectron on slurm
- bash rebuild_detectron2.sh

Download weights
- R-FPN ResNet50 trained with C-RBP for 20 steps: `wget https://bashupload.com/qyTMQ/-D4Ro.pth`
- R-FPN ResNet101 trained with C-RBP for 20 steps: `wget https://bashupload.com/V_Rhr/lwUiz.pth`
- FPN ResNet50: `wget https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/model_final_dbfeb4.pkl`
- FPN ResNet101: `wget https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl`

Train a model
- R-FPN ResNet50 trained with C-RBP for 20 steps: `python tools/train_net.py --num-gpus 10 --config-file configs/COCO-PanopticSegmentation/panoptic_rfpn_R_50_cbp20_1x.yaml SOLVER.IMS_PER_BATCH 40 SOLVER.BASE_LR 0.05`

Test a model
- R-FPN ResNet50 trained with C-RBP for 20 steps: `python tools/train_net.py --eval-only --num-gpus 10 --config-file configs/COCO-PanopticSegmentation/panoptic_rfpn_R_50_cbp20_1x-test.yaml MODEL.WEIGHTS <path-to-weights>`

---
Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Includes more features such as panoptic segmentation, densepose, Cascade R-CNN, rotated bounding boxes, etc.
* Can be used as a library to support [different projects](projects/) on top of it.
  We'll open source more research projects in this way.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
