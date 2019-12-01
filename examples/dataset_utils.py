import os
import pandas as pd
import cv2
import glob
from detectron2.structures import BoxMode


def get_annotated_dataset(root_dir, dataset_dirs):
    dataset_dicts = []

    for dataset_dir in dataset_dirs:
        annotations_df = pd.read_csv(
            os.path.join(root_dir, dataset_dir, "annotations.csv"),
            names=["filename", "frame_idx", "tlx", "tly", "brx", "bry", "class"])
        annotations_per_file = annotations_df.groupby(annotations_df.filename)

        for filename, bbs in annotations_per_file:
            record = {}
            record["file_name"] = os.path.join(root_dir, filename)
            im = cv2.imread(record["file_name"], cv2.IMREAD_COLOR)
            record["height"], record["width"] = im.shape[:2]

            record["annotations"] = []
            for _, row in bbs.iterrows():
                bb = row[["tlx", "tly", "brx", "bry"]].to_numpy(dtype=float).tolist()
                record["annotations"].append({
                    "bbox": bb,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": row["class"],
                    "iscrowd": 0
                })
            dataset_dicts.append(record)

    return dataset_dicts
