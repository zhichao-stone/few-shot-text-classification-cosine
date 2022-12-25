import torch
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

label_to_prompt = [
    '通信 网络 天线',       # 0
    '农业 植物 养殖',       # 1
    '制造 机械 装置',       # 2
    '数据处理 数据存储',    # 3
    '物联 区块链 交通',     # 4
    '化学 材料 工艺',       # 5
    '深度学习 算法 智能',   # 6
    '生态 环保 废弃处理',   # 7
    '疾病 护理 医疗',       # 8
    '图像 显示 视频',       # 9
    '光电 电子 元件',       # 10
    '焊接 机械 自动',       # 11
    '电极 金属 冶金',       # 12
    '电力 输电 电缆',       # 13    供电输电高度相关
    '光伏组件 太阳能',      # 14    太阳能发电高度相关
    '电动汽车 车用电池',    # 15    电动车辆高度相关
    '无机 晶体 涂层',       # 16
    '风力 风机 风电',       # 17    风力发电高度相关
    '药物 合成 制药',       # 18
    '复合 纳米 超导',       # 19    纳米超导材料高度相关
    '材料的测定与试验',     # 20
    '车辆 铁路 列车',       # 21
    '聚合 纺织 纤维',       # 22
    '航天 定位 导航',       # 23
    '水利 检测 测量',       # 24
    '洗砂 烧结 锅炉',       # 25
    '水产 加工 食品',       # 26
    '核技术 反应堆 核电',   # 27    核电高度相关
    '开采 船舶 海洋',       # 28
    '航天 航空 飞行',       # 29
    '信息 推送 广告',       # 30    广告推送高度相关
    '家居 住宅 建筑 '       # 31
]

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
        analyze_each_label(labels, predictions)

    return loss, f1_macro, f1_micro

def analyze_each_label(labels, preds):
    '''
    计算每个类别的准确率和F1
    
    labels: 1D List，真实样本所属类别
    preds: 1D List，模型预测结果
    '''
    each_correct = [0] * 32
    each_total = [0] * 32
    for i, label in enumerate(labels):
        each_total[label] += 1
        if label == preds[i]:
            each_correct[label] += 1
    
    each_f1 = [0]*32
    for i in range(32):
        tp, fp, fn = 0, 0, 0
        for j, label in enumerate(labels):
            p = preds[j]
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
    print('# ACC, F1 of each label:')
    for i in range(32):
        print(f'Label {i:2d} : acc={each_correct[i]/each_total[i]:.1%} | f1={each_f1[i]:.1%} | total={each_total[i]}')
