import os, json, random
import argparse
import logging
import numpy as np
import torch

class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Code for CCF Few-shot Data Classification')

        parser.add_argument('--num_class', type=int, default=32, help='num of classes')
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--max_len', type=int, default=512, help='max length of sentence')
        parser.add_argument('--hidden_size', type=int, default=768, help='dimension of hidden uits in BiLSTM layer ')
        parser.add_argument('--num_layers', type=int, default=1, help='num of BiLSTM layer')
        parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
        parser.add_argument('--cuda', type=int, default=0, help='num of gpu')
        # Data configs
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--train_file', type=str, default='train.json')
        parser.add_argument('--test_file', type=str, default='test.json')
        parser.add_argument('--dev_ratio', type=float, default=0.1, help='split 10 percentages of training data as validation')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        # save model configs
        parser.add_argument('--model_name', type=str, default='CCF_Model')
        parser.add_argument('--save_path', type=str, default='./save')
        parser.add_argument('--best_score', type=float, default=0.0, help='save checkpoint if mean_f1 > best_score')
        parser.add_argument('--alpha', type=float, default=0.6, help='alpha * cosine(title) + (1-alpha) * cosine(text)')
        parser.add_argument('--method', type=str, default='Cosine', help='use Baseline or Cosine')
        # learning configs
        parser.add_argument('--max_epochs', type=int, default=20, help='max epochs')
        parser.add_argument('--print_steps', type=int, default=50, help='number of steps to log training metrics')
        parser.add_argument('--warmup_steps', type=int, default=200, help="warm ups for parameters not in bert or vit")
        parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total steps to run')
        parser.add_argument('--minimum_lr', type=float, default=1e-6, help='minimum learning rate')
        parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
        parser.add_argument('--adam_epsilon', default=1e-6, type=float, help='Epsilon for Adam optimizer')
        # title BERT
        parser.add_argument('--bert_dir', type=str, default='chinese-roberta-wwm-ext')
        parser.add_argument('--bert_learning_rate', type=float, default=5e-5)

        args = parser.parse_args()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])
        
        method_options = ['Baseline', 'Cosine', 'WeightedBERT']
        assert self.method in method_options

        # device
        self.device = torch.device('cuda:{}'.format(self.cuda)) if self.cuda >= 0 and torch.cuda.is_available() else torch.device('cpu')
        # save model configs
        self.model_dir = f'{self.save_path}/{self.model_name}'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # data configs
        self.train_path = f'{self.data_dir}/{self.train_file}'
        self.test_path = f'{self.data_dir}/{self.test_file}'

        # set random seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)  # set seed for cpu
        torch.cuda.manual_seed(self.seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(self.seed)  # set seed for all gpu
        
        # set logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

        # save config
        with open(os.path.join(self.model_dir, 'config.json'), 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)
    
    def __str__(self):
        ret = ''
        for key in self.__dict__:
            ret = ret + f'{key} : {self.__dict__[key]}\n'
        return ret


if __name__ == '__main__':
    config = Config()
    print(config)