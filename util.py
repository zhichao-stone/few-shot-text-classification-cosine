import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

def build_optimizer(model, config):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and not any(nd in i for nd in no_decay))],
         'lr': config.learning_rate, 'weight_decay': config.weight_decay},
        {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and any(nd in i for nd in no_decay))],
         'lr': config.learning_rate, 'weight_decay': 0.0},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and not any(nd in i for nd in no_decay))],
         'lr': config.bert_learning_rate, 'weight_decay': config.weight_decay},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and any(nd in i for nd in no_decay))],
         'lr': config.bert_learning_rate, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps)

    return optimizer, scheduler


# EMA
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def evaluate(model, loader, test_mode=False):
    predictions = []
    labels = []
    losses = []

    with torch.no_grad():
        model.eval()
        for batch in loader:
            loss, acc, pred, label = model(batch)
            loss = loss.mean()

            predictions.extend(pred.cpu().detach().numpy())
            labels.extend(label.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

    loss = sum(losses) / len(losses)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')

    if test_mode:
        each_correct = [0] * 32
        each_total = [0] * 32
        for i, label in enumerate(labels):
            label = int(label)
            each_total[label] += 1
            if label == predictions[i]:
                each_correct[label] += 1
        
        each_f1 = [0]*32
        for i in range(32):
            tp, fp, fn = 0, 0, 0
            for j, label in enumerate(labels):
                p = predictions[j]
                if p == i:
                    if label == i:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if label == i:
                        fn += 1
            precision = tp / (tp + fp + 1e-5)
            recall = tp / (tp + fn + 1e-5)
            each_f1[i] = 2 * precision * recall / (precision + recall + 1e-5)
        print('# Acc of each label:')
        for i in range(32):
            print(f'Label {i:2d} : acc={each_correct[i]/each_total[i]:.1%} | f1={each_f1[i]:.1%} | total={each_total[i]}')

    return loss, f1_macro, f1_micro
