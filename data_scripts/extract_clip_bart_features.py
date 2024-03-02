import os
from PIL import Image
from tqdm import tqdm
import argparse
import pathlib

import torch
import clip

from load_aokvqa import load_aokvqa, get_coco_path
from transformers import BartTokenizer, BartModel

parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, default='/home/huangsiyong/data/aokvqa/', required=False, dest='aokvqa_dir')
parser.add_argument('--coco-dir', type=pathlib.Path, default='/home/huangsiyong/data/coco2017/', required=False, dest='coco_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--model-type', type=str, choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], required=True, dest='model_type')
parser.add_argument('--out', type=pathlib.Path, required=True, dest='output_file')
args = parser.parse_args()

assert args.output_file.suffix == '.pt'

## Load dataset

dataset = load_aokvqa(args.aokvqa_dir, args.split)

## Load model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.model_type, device=device)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

## Encoding loop

with torch.no_grad():
    embeddings = {}
    for d in tqdm(dataset):
        img = Image.open(get_coco_path(args.split, d['image_id'], args.coco_dir))
        img = preprocess(img).unsqueeze(0).to(device)
        image_features = model.encode_image(img)  # [1, 512]
        q = d["question"]  # What type of race is this?
        choices = d['choices']
        qa_list = []
        bart_ids = []
        for c in choices:
            qa = 'Question: ' + q + ' Answer: ' + c  # Question: What is the boy on the right holding? Answer: mace
            qa_text = clip.tokenize(qa).to(device)  # [1, 77] [[49406, 4382(Question), 281(:), 8 tokens, 286(?), 4518(Answer), 281(:), 44869, 49407, 0]
            qa_text_features = model.encode_text(qa_text)  # [1, 512]
            qa_list.append(qa_text_features[0].float().cpu())
            bart_ids.append(tokenizer(qa, return_tensors="pt"))
        qa_list = torch.stack(qa_list, dim=0)
        embeddings[d['question_id']] = {
            'qa_list': qa_list,
            'image': image_features[0].float().cpu(),
            'bart_inputs': bart_ids
        }

    torch.save(embeddings, args.output_file)
