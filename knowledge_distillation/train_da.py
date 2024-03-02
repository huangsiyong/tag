import os
import sys

sys.path.append("..")
import json
import argparse
import pathlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import sentencepiece;
import pytorch_lightning as pl

import torchmetrics.functional as MF
from transformers import BartTokenizer, BartModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    BartForConditionalGeneration

from load_aokvqa import load_aokvqa


def debug():
    main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, default='/home/huangsiyong/data/aokvqa/', required=False,
                        dest='aokvqa_dir')
    parser.add_argument('--vocab', type=argparse.FileType('r'),
                        default='/home/huangsiyong/data/aokvqa/large_vocab_train.csv', required=False)
    parser.add_argument('--log-dir', type=pathlib.Path, default='/home/huangsiyong/ali/aokvqa/logs/', dest='log_dir',
                        required=False)
    #
    parser.add_argument('--backbone', type=str, choices=['clip', 'resnet', 'bert'], default='clip', required=False)
    parser.add_argument('--clip-model-type', type=str,
                        choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14',
                                 'ViT-L/14@336px'],
                        dest='clip_model_type', default='ViT-L/14@336px', required=('clip' in sys.argv))
    parser.add_argument('--train-features', type=pathlib.Path,
                        default='/home/huangsiyong/ali/aokvqa/features/DA_train', required=False,
                        dest='train_features')
    parser.add_argument('--val-features', type=pathlib.Path,
                        default='/home/huangsiyong/ali/aokvqa/features/DA_val', required=False,
                        dest='val_features')
    parser.add_argument('--vocab-features', type=pathlib.Path,
                        default='/home/huangsiyong/ali/aokvqa/features/clip-ViT-B-32_large_vocab.pt', required=False,
                        dest='vocab_features')
    #
    parser.add_argument('--objective', type=str, choices=['classifier', 'contrastive'], default='contrastive',
                        required=False)
    parser.add_argument('--inputs', nargs='+', type=str, choices=['question', 'image'], default='image', required=False)
    # Defaults
    parser.add_argument('--bs', type=int, default=2, dest='batch_size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--task', type=str, choices=['MC', 'DA'], default='DA', required=False)
    args = parser.parse_args()

    pl.seed_everything(1)
    vocab = args.vocab.read().splitlines()

    ## Data loading

    dm = AokvqaEmbeddingsDataModule(
        args.aokvqa_dir,
        args.train_features,
        args.val_features,
        args.objective,
        args.backbone,
        args.inputs,
        vocab,
        args.vocab_features,
        batch_size=args.batch_size,
        num_workers=16
    )

    aokvqa_set = load_aokvqa(args.aokvqa_dir, 'train')
    len_train = len(aokvqa_set)
    steps_per_epoch = len_train // args.batch_size
    print("len_train, steps_per_epoch: ", len_train, steps_per_epoch)
    ## Model definition

    model = LinearClassifier(
        args.objective,
        args.backbone,
        args.clip_model_type,
        args.inputs,
        len(vocab),
        args.lr,
        args.epochs,
        steps_per_epoch
    )
    ## Training and testing loops
    logger = pl.loggers.TensorBoardLogger(
        args.log_dir,
        name=args.name
    )

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.gpus,
        max_epochs=args.epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                filename="{epoch:02d}-{val_acc:.2f}",
                mode="max"
            )
        ],
    )

    trainer.fit(model, dm)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class AokvqaEmbeddingsDataset(Dataset):
    def __init__(self, aokvqa_dir, split, input_features, objective, backbone, inputs, vocab, vocab_features):
        self.aokvqa_set = load_aokvqa(aokvqa_dir, split)
        self.path = input_features
        self.objective = objective
        self.vocab_len = len(vocab)

    def __getitem__(self, index):
        o = self.aokvqa_set[index]
        q = o['question_id']
        embedding = torch.load(self.path / (q + ".pt"))
        gt = embedding['da_target']
        i = embedding['image']
        t = embedding['da_list']
        bart_input = embedding['da_bart_inputs']
        encoder_input = embedding['encoder_input']
        decoder_input = embedding['rationale']
        answer_idx = embedding['answer_index']
        return i, t, gt, bart_input, encoder_input, decoder_input, answer_idx

    def __len__(self):
        return len(self.aokvqa_set)


def aokvqa_collate_fn(batch):
    image = []
    text = []
    gt = []
    answer_idx = []
    ids = []
    mask = []
    encoder_input = []
    encoder_mask = []
    decoder_input = []
    decoder_mask = []
    for b in batch:
        i, t, c, bart_input, e, d, a = b
        image.append(i)
        text.append(t)
        gt.append(torch.tensor(c))
        answer_idx.append(torch.tensor(a))

        encoder_input.append(e["input_ids"].squeeze(0))
        encoder_mask.append(e["attention_mask"].squeeze(0))
        decoder_input.append(d["input_ids"].squeeze(0))
        decoder_mask.append(d["attention_mask"].squeeze(0))

        for p in bart_input:
            ids.append(p["input_ids"].squeeze(0))
            mask.append(p["attention_mask"].squeeze(0))
    image = torch.stack(image)
    text = torch.stack(text, dim=0)
    gt = torch.stack(gt)
    answer_idx = torch.stack(answer_idx)

    ids = pad_sequence(ids, batch_first=True, padding_value=1)
    mask = pad_sequence(mask, batch_first=True)

    encoder_input = pad_sequence(encoder_input, batch_first=True, padding_value=1)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True)
    decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=1)
    decoder_mask = pad_sequence(decoder_mask, batch_first=True)

    return image, text, gt, ids.reshape(-1, 10, ids.size(-1)), mask.reshape(-1, 10, mask.size(-1)), \
           encoder_input, encoder_mask, decoder_input, decoder_mask, answer_idx


class AokvqaEmbeddingsDataModule(pl.LightningDataModule):

    def __init__(self, aokvqa_dir, train_features, val_features, objective, backbone, inputs, vocab, vocab_features,
                 batch_size=1, num_workers=0):
        super().__init__()
        self.aokvqa_dir = aokvqa_dir
        self.train_features = train_features
        self.val_features = val_features
        self.objective = objective
        self.backbone = backbone
        self.inputs = inputs
        self.vocab = vocab
        self.vocab_features = vocab_features
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = AokvqaEmbeddingsDataset(
            self.aokvqa_dir, 'train', self.train_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features
        )
        self.val_dataset = AokvqaEmbeddingsDataset(
            self.aokvqa_dir, 'val', self.val_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=int(0.8 * self.num_workers),
            persistent_workers=True,
            collate_fn=aokvqa_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
            num_workers=int(0.2 * self.num_workers),
            persistent_workers=True,
            collate_fn=aokvqa_collate_fn
        )


class LinearClassifier(pl.LightningModule):
    def __init__(self, objective, backbone, clip_model_type, inputs, vocab_len, lr=0.001, epochs=100,
                 steps_per_epoch=100):
        super().__init__()
        self.save_hyperparameters(ignore=['lr'])
        self.lr = lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        if self.hparams.backbone == 'clip':
            clip_dim = {
                'RN50': 1024,
                'RN50x4': 640,
                'RN50x16': 768,
                'RN50x64': 1024,
                'RN101': 512,
                'ViT-B/32': 512,
                'ViT-B/16': 512,
                'ViT-L/14': 768,
                'ViT-L/14@336px': 768,
            }[clip_model_type]
            emb_dim = clip_dim * len(inputs)
        elif self.hparams.backbone == 'resnet':
            emb_dim = 2048
        elif self.hparams.backbone == 'bert':
            emb_dim = 768

        if self.hparams.objective == 'classifier':
            out_dim = vocab_len
        elif self.hparams.objective == 'contrastive':
            out_dim = clip_dim

        # bart_model = BartForConditionalGeneration.from_pretrained('/home/huangsiyong/model_zoo/huggingface/bart_base')
        bart_model = BartForConditionalGeneration.from_pretrained('/home/huangsiyong/model_zoo/huggingface/bart_large')
        self.encoder = bart_model.model.encoder
        self.decoder = bart_model.model.decoder
        self.lm_head = bart_model.lm_head

        self.visual_linear = nn.Linear(out_dim, 1024)
        self.linear0 = nn.Linear(1024, out_dim)

    def forward(self, ids, mask):
        ori_feature = self.encoder(ids, mask).last_hidden_state
        mean_feature = self.mean(ori_feature, mask)
        feature = self.linear0(mean_feature)
        return feature, ori_feature

    def forward_decode(self, image_feature, encoder_input, encoder_mask, decoder_input, decoder_mask):
        text_feature = self.encoder(encoder_input, encoder_mask).last_hidden_state

        # encoder_outputs = torch.cat(image_feature, text_feature, dim=1)
        # attention_mask = encoder_mask

        encoder_v = self.visual_linear(image_feature)
        encoder_outputs = torch.cat((encoder_v.unsqueeze(1), text_feature), dim=1)
        attention_mask = torch.cat((encoder_mask[:, 0].unsqueeze(1), encoder_mask), dim=1)

        input_ids = shift_tokens_right(decoder_input, 1, 2)
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        )
        lm_logits = self.lm_head(outputs[0])
        masked_lm_loss = F.cross_entropy(lm_logits.view(-1, 50265), decoder_input.view(-1))
        return masked_lm_loss

    def mean(self, x, mask):
        dim = x.size(-1)
        x_after = x * mask.unsqueeze(-1).repeat(1, 1, dim)
        x_sum = x_after.sum(1)
        mask_sum = mask.sum(1)
        y = x_sum / mask_sum.unsqueeze(-1).repeat(1, dim)
        return y

    def loss_func1(self, i2t, gt_after):
        # contrastive loss between i and t'
        return F.binary_cross_entropy(i2t, gt_after)

    def loss_func2(self, t_norm, t):
        # mse loss between t and t'
        return F.mse_loss(t_norm, t)

    def loss_func3(self, t, t_norm):
        # contrastive loss between t' and t
        t2t = ((t_norm @ t.T) / 0.02).softmax(dim=-1)
        t2t_gt = torch.arange(0, t.shape[0], dtype=torch.int64, device=self.device)
        return F.cross_entropy(t2t, t2t_gt)

    def loss_func4(self, i, t_norm, gt_after):
        # contrastive loss between t'(1/4) and i
        t_gt = t_norm[gt_after]
        t2i = ((t_gt @ i.T) / 0.02).softmax(dim=-1)
        indices = torch.arange(0, i.shape[0], dtype=torch.int64, device=self.device)
        return F.cross_entropy(t2i, indices)

    def get_mc_acc(self, i2t, gt_after):
        idx = i2t.size(-1) - i2t.flip(dims=[0]).argmax().item() - 1
        if idx == gt_after[0]:
            return 1.0
        else:
            return 0.0

    def compute_loss(self, batch):
        i, t, gt, ids, mask, encoder_input, encoder_mask, decoder_input, decoder_mask, answer_idx = batch
        ids = ids.reshape(-1, ids.size(-1))
        mask = mask.reshape(-1, mask.size(-1))
        t_after, t_ori = self.forward(ids, mask)
        t = t.reshape(-1, t.size(-1))
        t_norm = F.normalize(t_after, dim=-1)
        i2t = ((i @ t_norm.T) / 0.02).softmax(dim=-1)
        gt = gt*1.0 / 10
        gt_for_loss1 = torch.zeros(i2t.size(0), i2t.size(1), device=self.device)
        for b in range(i.size(0)):
            gt_for_loss1[b, b*10:(b+1)*10] = gt[b]

        matrix = torch.triu(torch.ones((answer_idx.size(0), answer_idx.size(0)), device=self.device), diagonal=1) * 10
        gt_for_loss2 = torch.tensor((answer_idx + matrix.sum(0)), dtype=torch.int64)
        if i.size(0) == 1:
            loss4 = 0.0
        else:
            loss4 = self.forward_decode(i, encoder_input, encoder_mask, decoder_input, decoder_mask)

        loss1 = self.loss_func1(i2t, gt_for_loss1)
        # loss1 = 0.0
        # loss2 = 1000 * self.loss_func2(t_norm, t)
        loss2 = self.loss_func4(i, t_norm, gt_for_loss2)
        # loss2 = 0.0
        loss3 = self.loss_func3(t, t_norm)
        # loss3 = 0.0

        loss = 1.0 * loss1 + 1.0 * loss2 + 2.0 * loss3 + 0.1 * loss4
        # acc
        indices = torch.arange(0, i.shape[0], dtype=torch.int64, device=self.device)
        if i.size(0) == 1:
            acc = self.get_mc_acc(i2t, gt_for_loss2)
        else:
            acc = torch.mean(i2t[indices, gt_for_loss2])
        return loss, acc, loss1, loss2, loss3, loss4

    def training_step(self, batch, batch_idx):
        loss, acc, loss1, loss2, loss3, loss4 = self.compute_loss(batch)
        self.log("train_loss", loss)
        self.log("train_loss/loss1", loss1)
        self.log("train_loss/loss2", loss2)
        self.log("train_loss/loss3", loss3)
        self.log("train_loss/loss4", loss4)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, loss1, loss2, loss3, loss4 = self.compute_loss(batch)
        self.log("val_loss", loss)
        self.log("val_loss/loss1", loss1)
        self.log("val_loss/loss2", loss2)
        self.log("val_loss/loss3", loss3)
        self.log("val_loss/loss4", loss4)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        for name, p in self.encoder.named_parameters():
            if name.startswith('layers.11') is False:
                p.requires_grad = False
        encoder_params = filter(lambda x: x.requires_grad is not False, self.encoder.parameters())
        for name, p in self.decoder.named_parameters():
            if name.startswith('layers.11') is False:
                p.requires_grad = False
        decoder_params = filter(lambda x: x.requires_grad is not False, self.decoder.parameters())
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': self.lr * 0.1},
            {'params': decoder_params, 'lr': self.lr * 0.1},
            {'params': self.linear0.parameters(), 'lr': self.lr},
            {'params': self.visual_linear.parameters(), 'lr': self.lr},
        ], weight_decay=0)

        steps = self.epochs * self.steps_per_epoch
        warmup_steps = 1 * self.steps_per_epoch
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps, num_cycles=0)

        if optimizer and scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif optimizer:
            return [optimizer]


if __name__ == '__main__':
    main()
