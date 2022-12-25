import json
import random
from util import label_to_prompt

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer


class CCFDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.max_len = config.max_len
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.data = data
        self.method = config.method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        title = self.data[index]['title']
        assignee = self.data[index]['assignee']
        abstract = self.data[index]['abstract']
        
        if self.method == 'Baseline':
            text = f'这份专利的标题为《{title}》，由\"{assignee}\"申请，详细说明为：{abstract}'
            text_inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True)
            text_inputs['input_ids'][0] = 101
            text_inputs = {k: torch.LongTensor(v) for k, v in text_inputs.items()}
            d = {
                'text_inputs': text_inputs['input_ids'],
                'text_mask': text_inputs['attention_mask'],
                'text_type_ids': text_inputs['token_type_ids'],
                'label': torch.LongTensor([self.data[index]['label_id']])
            }
        elif self.method == 'Cosine':
            text = f'专利由\"{assignee}\"申请，详细说明为：{abstract}'
            text_inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True)
            title_inputs = self.tokenizer(title, max_length=30, padding='max_length', truncation=True)
            text_inputs['input_ids'][0] = 101
            title_inputs['input_ids'][0] = 101
            text_inputs = {k: torch.LongTensor(v) for k, v in text_inputs.items()}
            title_inputs = {k: torch.LongTensor(v) for k, v in title_inputs.items()}
            d = {
                'text_inputs': text_inputs['input_ids'],
                'text_mask': text_inputs['attention_mask'],
                'text_type_ids': text_inputs['token_type_ids'],
                'title_inputs': title_inputs['input_ids'],
                'title_mask': title_inputs['attention_mask'],
                'title_type_ids': title_inputs['token_type_ids'],
                'label': torch.LongTensor([self.data[index]['label_id']])
            }

        return d

def create_dataloaders(config):
    dev_ratio = config.dev_ratio
    data = []
    with open(config.train_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data.append(json.loads(line))
    
    random.shuffle(data)
    train_data = data[int(dev_ratio * len(data)):]
    dev_data = data[:int(dev_ratio * len(data))]

    train_dataset = CCFDataset(train_data, config)
    dev_dataset = CCFDataset(dev_data, config)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset),
        drop_last=True
    )
    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=config.batch_size,
        sampler=SequentialSampler(dev_dataset),
        drop_last=False
    )

    return train_loader, dev_loader
