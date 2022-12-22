import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from patent_classification import label_to_prompt


class CCFModel(nn.Module):
    def __init__(self, config):
        super(CCFModel, self).__init__()
        self.device = config.device
        self.dropout = config.dropout
        self.method = config.method

        if self.method == 'Baseline':
            self.bert = BertModel.from_pretrained(config.bert_dir, output_hidden_states=False)
        else:
            self.bert = BertModel.from_pretrained(config.bert_dir, output_hidden_states=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.text_embedding = self.bert.embeddings
        self.fc = nn.Linear(768 * 4, config.num_class)
        self.alpha = config.alpha

    def _get_label_feature(self) -> torch.Tensor:
        inputs_id, mask, type_id = [], [], []
        for label in label_to_prompt:
            inputs = self.tokenizer(label, max_length=11, padding='max_length', truncation=True)
            inputs[0] = 101
            inputs_id.append(inputs['input_ids'])
            mask.append(inputs['attention_mask'])
            type_id.append(inputs['token_type_ids'])
        inputs_id, mask, type_id = torch.LongTensor(inputs_id).cuda(), torch.LongTensor(mask).cuda(), torch.LongTensor(type_id).cuda()
        embeds = self.text_embedding(inputs_id, type_id)
        out = self.bert(attention_mask=mask, inputs_embeds=embeds)
        return out[0][:, 0, :]
    
    def forward(self, data):
        text_embeds = self.text_embedding(data['text_inputs'].cuda(), data['text_type_ids'].cuda())
        text_out = self.bert(attention_mask=data['text_mask'].cuda(), inputs_embeds=text_embeds)
        label = data['label'].cuda()

        if self.method == 'Baseline':
            # version 1 -- BERT pooling
            hidden_states = text_out.hidden_states[-4:]
            hidden_states = [i.mean(dim=1) for i in hidden_states]
            out = self.fc(torch.cat(hidden_states, dim=1))
            loss, acc, pred, label = self.cal_loss(out, label)

        elif self.method == 'Cosine':
            # version 2 -- cosine
            title_embeds = self.text_embedding(data['title_inputs'].cuda(), data['title_type_ids'].cuda())
            title_out = self.bert(attention_mask=data['title_mask'].cuda(), inputs_embeds=title_embeds)
            text_out = text_out[0][:, 0, :].unsqueeze(dim=1).repeat(1, 32, 1)
            title_out = title_out[0][:, 0, :].unsqueeze(dim=1).repeat(1, 32, 1)
            label_out = self._get_label_feature().unsqueeze(dim=0).repeat(text_out.shape[0], 1, 1)
            out = torch.add(self.alpha * torch.cosine_similarity(title_out, label_out, dim=2), \
                    (1 - self.alpha) * torch.cosine_similarity(text_out, label_out, dim=2))
            loss, acc, pred, label = self.cal_mse_loss(out, label)

        return loss, acc, pred, label


    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
    @staticmethod
    def cal_mse_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss_label = F.one_hot(label, num_classes=32).to(torch.float)
        loss = F.mse_loss(prediction, loss_label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
        