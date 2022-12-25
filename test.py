import json
from torch.utils.data import SequentialSampler, DataLoader
from util import evaluate

from config import Config
from data_helper import CCFDataset
from model import load_model

config = Config()
# read test data
test_data = []
with open('data/test.json', 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        test_data.append(json.loads(line))
dataset = CCFDataset(test_data, config)
loader = DataLoader(
    dataset=dataset, batch_size=config.batch_size, 
    sampler=SequentialSampler(dataset), drop_last=False
)
# load model
dev_f1_macro, model = load_model(config, config.model_name, config.method)
# test
test_loss, test_f1_macro, test_f1_micro = evaluate(model, loader, test_mode=True)
print(f'On Dev  : macro f1 {dev_f1_macro:.4f}')
print(f'On Test : loss {test_loss:.4f}, macro f1 : {test_f1_macro:.4f} , micro f1 : {test_f1_micro:.4f}')