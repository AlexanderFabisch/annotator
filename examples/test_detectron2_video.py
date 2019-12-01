import sys
PATH_TO_DETECTRON2 = "detectron2"
sys.path.append(PATH_TO_DETECTRON2)
import argparse
import yaml
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tqdm
from examples.dataset_utils import get_annotated_dataset


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if "classes" not in config:
        raise Exception("Could not find class names")
    n_classes = len(config["classes"])
    classes = config["classes"]

    cfg = get_cfg()
    cfg.merge_from_file(args.model_config)
    cfg.DATASETS.TRAIN = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 50000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    if args.model_weights is None:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("custom_test",)

    predictor = DefaultPredictor(cfg)

    DatasetCatalog.register("custom_test", lambda d="test": None)
    MetadataCatalog.get("custom_test").set(thing_classes=classes)
    custom_metadata = MetadataCatalog.get("custom_test")

    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vis = VideoVisualizer(metadata=custom_metadata)
    for i in tqdm.tqdm(range(0, n_frames, args.skip_frames)):
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = cap.read()
        assert success
        outputs = predictor(image)
        v = vis.draw_instance_predictions(
            image[:, :, ::-1], outputs["instances"].to("cpu"))
        filename = os.path.join(args.output, "prediction_%09d.jpg" % i)
        cv2.imwrite(filename, v.get_image()[:, :, ::-1])
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Annotator")
    parser.add_argument("video", help="Video file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--skip_frames", default=1, type=int, help="Skip N frames")
    parser.add_argument("--model_weights", default=None, help="Model weights.")
    parser.add_argument(
        "--model_config",
        default=os.path.join(PATH_TO_DETECTRON2, "configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"),
        help="Model configuration file.")
    parser.add_argument(
        "--config", nargs="?", default="config.yaml",
        help="Configuration file for annotator")
    return parser.parse_args()


def cv2_imshow(im):
    plt.figure(figsize=(16, 9))
    plt.imshow(im[:, :, (2, 1, 0)])


if __name__ == "__main__":
    main()
