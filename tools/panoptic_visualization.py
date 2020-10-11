# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import numpy as np
np.random.seed(42)
import os
from itertools import chain
import cv2
import tqdm
from PIL import Image
from glob import glob
from skimage import io

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor


def setup(args):
    cfg = get_cfg()
    # if args.config_file:
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.ckpt
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        # required=True,
        default="dataloader",
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--ckpt", default="./", help="path to model checkpoint")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("--neural", action="store_true", help="grab activities for neural analysis")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of image batches to viz.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    num_batches = args.num_batches
    neural = args.neural

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            # print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    if neural:
        neural_im_list = np.load(os.path.join('datasets', 'coco_image_id.npy'))
    scale = 1.0
    if args.source == "dataloader":
        test_data_loader = build_detection_test_loader(cfg, dataset_name='coco_2017_val_panoptic_separated')
        count = 0
        if neural:
            ims = glob(os.path.join("datasets", "bold5000", "*"))
            ims = [im for im in ims if 'COCO' not in im]
            for imf in tqdm.tqdm(ims, total=len(ims), desc="Neural image batches"):
                oname = imf.split(os.path.sep)[-1]
                im = io.imread(imf)
                res = predictor(im, neural=neural)  # noqa
                panoptic_seg, segments_info = res["panoptic_seg"]
                activities = res["activations"]
                v = Visualizer(im[:, :, ::-1], metadata, scale=scale)
                try:
                    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
                    output(v, '%s.jpg' % oname)
                except:
                    pass
                    # io.imwrite(os.path.join(dirname, '%s-seg.jpg' % oname), panoptic_seg)
                    # io.imwrite(os.path.join(dirname, '%s-im.jpg' % oname), im[..., -1])
                # output(v, "%s.jpg".format(oname))
                np.savez(os.path.join(dirname, oname), **activities)
        else:
            for batch in tqdm.tqdm(test_data_loader, total=num_batches, desc='COCO image batches'):
                for per_image in batch:
                    im = per_image["image"]
                    try:
                        im = per_image["image"]
                        im = im.cpu().numpy().transpose((1, 2, 0))
                        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                        v = Visualizer(im[:, :, ::-1], metadata, scale=scale)
                        v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
                        output(v, str(per_image["image_id"]) + ".jpg")
                    except Exception as e:
                        print("Failed on {}: {}".format(str(per_image["image_id"]), e))
                count += 1
                if count >= num_batches:
                    break
    else:
        raise NotImplementedError
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        if cfg.MODEL.KEYPOINT_ON:
            dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm.tqdm(dicts):
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))

