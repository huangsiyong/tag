import os
import json


def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    f = open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json"))
    dataset = json.load(f)
    f.close()
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

def get_coco2014_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2014", f"COCO_val2014_{image_id:012}.jpg")
