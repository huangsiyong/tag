import sys
import os
import argparse
import pathlib
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import numpy

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import sentencepiece; import pytorch_lightning as pl; import clip

from knowledge_distillation.train_decoder import LinearClassifier
from load_aokvqa import load_aokvqa
# from evaluation.remap_predictions import map_to_choices


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--features', type=pathlib.Path, required=True)
parser.add_argument('--out', type=argparse.FileType('w'), dest='output_file')
#
parser_weights = parser.add_mutually_exclusive_group(required=True)

parser_weights.add_argument('--ckpt', type=pathlib.Path, dest='checkpoint_path')

parser_weights.add_argument('--zero-shot', action='store_true', dest='clip_zero_shot')
parser.add_argument('--inputs', nargs='+', type=str, choices=['question', 'image'], required=('--zero-shot' in sys.argv))

parser.add_argument('--clip-model-type', type=str,
                    choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
                    dest='clip_model_type', required=('--zero-shot' in sys.argv and '--mc' in sys.argv))
#
args = parser.parse_args()


## Load dataset

aokvqa_set = load_aokvqa(args.aokvqa_dir, args.split)

## Load models

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.checkpoint_path is not None:
    classifier = LinearClassifier.load_from_checkpoint(args.checkpoint_path)
    classifier.to(device)
    hp = classifier.hparams
elif args.clip_zero_shot:
    classifier = nn.Identity().to(device)
    hp = pl.utilities.AttributeDict(backbone='clip', clip_model_type=args.clip_model_type, objective='zero-shot', inputs=args.inputs)

# Load input features

# embeddings = torch.load(args.features)
# if hp.backbone == 'clip':
#     for q in embeddings.keys():
#         embeddings[q]['qa_list'] /= embeddings[q]['qa_list'].norm(dim=-1, keepdim=True)
#         embeddings[q]['image'] = embeddings[q]['image'] / embeddings[q]['image'].norm(dim=-1, keepdim=True)

# Load vocab, vocab features, clip

if hp.objective in ['contrastive', 'zero-shot']:
        clip_model = clip.load(hp.clip_model_type, device=device)[0]
        logit_scale = clip_model.logit_scale.exp().cpu()

## Prediction loop

predictions = {}
results = {}
correct = 0
correct_2 = 0

with torch.no_grad():
    for o in tqdm(aokvqa_set):
        q = o['question_id']
        embedding = torch.load(args.features / (q + ".pt"))
        i = embedding['image']
        t = embedding['qa_list']
        bart_inputs = embedding['bart_inputs']
        encoder_input = embedding['encoder_input']

        i = i.view(1, -1).to(device)
        ids = encoder_input["input_ids"].to(device)
        mask = encoder_input["attention_mask"].to(device)
        print(o['question_id'])
        print(ids)
        classifier(ids, mask, i)


        # bart_inputs = embeddings[q]['bart_inputs']

        # ids = []
        # mask = []
        # for p in bart_inputs:
        #     ids.append(p["input_ids"].squeeze(0))
        #     mask.append(p["attention_mask"].squeeze(0))
        # ids = pad_sequence(ids, batch_first=True, padding_value=1)
        # mask = pad_sequence(mask, batch_first=True)
        # ids = ids.reshape(-1, ids.size(-1)).to(device)
        # mask = mask.reshape(-1, mask.size(-1)).to(device)
        #
        # t_after, t_ori = classifier(ids, mask)
        # t_norm = F.normalize(t_after, dim=-1).cpu()
        # x = (i.cpu() @ t_norm.t()) / 0.02
        # x = x.softmax(dim=-1)
        # predictions[q] = o['choices'][x.argmax().item()]
        # y = (i.cpu() @ t.t()) / 0.02
        # y = y.softmax(dim=-1)

        # if o['correct_choice_idx'] == x.argmax().item():
        #     correct += 1
        # if o['correct_choice_idx'] == y.argmax().item():
        #     correct_2 += 1

        # if o['correct_choice_idx'] != x.argmax().item() and o['correct_choice_idx'] != y.argmax().item():
        #     results[q] = {}
        #     results[q]['ID'] = o["image_id"]
        #     results[q]['CLP'] = y.detach().cpu().numpy().tolist()
        #     results[q]['TAG'] = x.detach().cpu().numpy().tolist()
        #     results[q]['answer'] = o['correct_choice_idx']

        # if args.split == 'test':
        #     correct = 0
        # elif o['correct_choice_idx'] == x.argmax().item():
        #     correct += 1
# print("acc:", correct/len(aokvqa_set))
# print("acc2:", correct_2/len(aokvqa_set))
# json.dump(predictions, args.output_file)
# json.dump(results, args.output_file)
