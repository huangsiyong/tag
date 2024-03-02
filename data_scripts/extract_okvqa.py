import os
from PIL import Image
from tqdm import tqdm
import argparse
import pathlib

import torch
import clip
import json
from load_aokvqa import load_aokvqa, get_coco2014_path

## Load dataset

# qid_to_topk = json.load(open('/home/huangsiyong/data/prophet/assets/candidates_okvqa.json'))
with open('/home/huangsiyong//data/prophet/datasets/okvqa/mscoco_val2014_annotations.json') as f:
    dataset = json.load(f)['annotations']
with open('/home/huangsiyong//data/prophet/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json') as f:
    Q = json.load(f)['questions']  # 'image_id': 581829, 'question': 'What fruit is typically added to the top of cereal?', 'question_id': 5818295

imgToQA = {ann['image_id']: [] for ann in dataset}
qa = {ann['question_id']: [] for ann in dataset}
qqa = {ann['question_id']: [] for ann in dataset}
for ann in dataset:
    imgToQA[ann['image_id']] += [ann]
    qa[ann['question_id']] = ann
for ques in Q:
    qqa[ques['question_id']] = ques

## Load model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device=device)

## Encoding loop

with torch.no_grad():
    embeddings = {}
    for d in tqdm(dataset):
        q_id = d['question_id']
        q = qqa[q_id]['question']  # What type of race is this?
        img = Image.open(get_coco2014_path('val', d['image_id'], '/home/huangsiyong/data/prophet/datasets/coco2014/'))
        img = preprocess(img).unsqueeze(0).to(device)
        image_features = model.encode_image(img)  # [1, 512]
        embeddings[d['question_id']] = {
            'question': q,
            'image': image_features[0].float().cpu(),
        }
    torch.save(embeddings, 'features/okvqa_val.pt')
