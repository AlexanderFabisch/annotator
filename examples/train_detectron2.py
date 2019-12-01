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
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
import numpy as np
import os
from examples.dataset_utils import get_annotated_dataset


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if "classes" not in config:
        raise Exception("Could not find class names")
    n_classes = len(config["classes"])
    classes = config["classes"]

    for d in ["train"]:
        DatasetCatalog.register("custom_" + d, lambda d=d: get_annotated_dataset(args.annotator_root, args.data_folders))
        MetadataCatalog.get("custom_" + d).set(thing_classes=classes)
    custom_metadata = MetadataCatalog.get("custom_train")

    cfg = get_cfg()
    cfg.merge_from_file(args.model_config)
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = args.initial_weights
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Annotator")
    parser.add_argument(
        "initial_weights", help="Initial model weights. Make sure to "
        "download the first initial weights from the model zoo: "
        "https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md")
    parser.add_argument("annotator_root", help="Root folder of annotator")
    parser.add_argument("data_folders", nargs="+", help="Subdirectory with labeled data")
    parser.add_argument(
        "--model_config",
        default=os.path.join(PATH_TO_DETECTRON2, "configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"),
        help="Model configuration file.")
    parser.add_argument(
        "--config", nargs="?", default="config.yaml",
        help="Configuration file for annotator")
    parser.add_argument(
        "--max_iter", type=int, default=1000,
        help="Maximum number of training iterations (including previous iterations)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
