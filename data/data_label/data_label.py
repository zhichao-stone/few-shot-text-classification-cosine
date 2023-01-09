label = [
    ['H01Q', 'H04(except H04M, H04N)', ], # 0 通信 网络 天线                            ** 通信网络高度相关
    ['A01', 'A23K', 'C05'], # 1 农业 植物 养殖
    ['G07'], # 2 制造 机械 装置
    ['G06F', 'G06K', 'G06Q', 'G11'], # 3 数据处理 数据存储
    ['G08G'], # 4 物联 区块链 交通
    ['B32'], # 5 化学 材料 工艺
    ['F24F', 'F25'], # 6 深度学习 算法 智能
    ['B09', 'B29', 'C02', 'C10'], # 7 生态 环保 废弃处理
    [], # 8 疾病 护理 医疗 
    ['G02(except F)', 'G06T', 'G09G', 'H04M', 'H04N'], # 9 图像 显示 视频
    ['G01R', 'G02F', 'H01(except F,M,Q)', 'H03', 'H05K'], # 10 光电 电子 元件
    ['B23(except H)', 'B24', 'B25', 'B26', 'B27', 'B65', 'F16', 'G05'], # 11 焊接 机械 自动    ** 自动化、机器人、机床加工相关
    ['B01J', 'B21', 'B22', 'B23H', 'C21', 'C22', 'C23', 'C25', 'H01F', 'H01M'], # 12 电极 金属 冶金
    ['H02B', 'H02G', 'H02H', 'H02J', 'H02M'], # 13 电力 输电 电缆                       ** 电缆供电高度相关
    ['H02S', 'F24S'], # 14 光伏组件 太阳能                                              ** 太阳能发电高度相关
    ['B60K', 'B60L', 'H02K'], # 15 电动汽车 车用电池                                    ** 电动汽车高度相关
    ['C01', 'C03', 'C04B', 'C09', 'C30', 'G16C'], # 16 无机 晶体 涂层
    ['F03D'], # 17 风力 风机 风电                                                       ** 风力发电高度相关
    ['A61K', 'A61P', 'C07D', 'C07K'], # 18 药物 合成 制药
    ['B82', 'C08F'], # 19 复合 纳米 超导                                                ** 纳米超导高度相关
    ['G01N'], # 20 材料的测定与试验
    # ['C21'], # 21 钢材 冶铁 生产
    ['B60(except K,L)', 'B61', 'B62', 'E01'], # 21 车辆 铁路 列车
    ['D'], # 22 聚合 纺织 纤维
    ['G01C', 'G01S'], # 23 航天 定位 导航
    ['E02', 'G01(except C,N,R,S,V)'], # 24 水利 检测 测量
    ['B03', 'B28', 'F22', 'F23', 'F27'], # 25 洗砂 烧结 锅炉
    ['A23(except A23K)', 'C08B', 'C08G', 'C11B', 'C12N'], # 26 水产 加工 食品
    ['G21'], # 27 核技术装置 核电                                               ** 核技术高度相关
    ['B63', 'B66D', 'E21B', 'F03B', 'G01V'], # 28 开采 船舶 海洋                        ** 海洋船舶高度相关
    ['B64'], # 29 航天 航空 飞行                                                        ** 航空航天高度相关
    ['G09F'], # 30 信息 推送 广告                                                        ** 广告推送高度相关
    ['E03', 'E04', 'E05', 'E06'] # 31 家居 住宅 建筑 
]

#### 核动力 > 广告 > 太阳能 光伏 > 风电 > 纤维 > 纳米
import json
with open('taxonomy.json', 'r', encoding='utf-8') as fr:
    ipc_to_label = json.load(fr)

def get_label(title: str, ipc: str):
    global ipc_to_label
    if '核电' in title or '核动力' in title or '铀' in title:
        return 27
    elif '广告' in title:
        return 30
    elif '太阳能' in title or '光伏' in title:
        return 14
    elif '风电' in title or '风力发电' in title:
        return 17
    elif '飞行器' in title or '飞行器' in title:
        return 29
    elif '电动车辆' in title or '电动汽车' in title:
        return 15
    elif '交通' in title or '物联网' in title or '区块链' in title:
        return 4
    elif '纳米' in title:
        return 19
    else:
        ans = ipc_to_label
        cnt = 0
        for s in ipc:
            if cnt >= 4:
                return -1
            if s in ans:
                ans = ans[s]
            else:
                if 'other' in ans:
                    ans = ans['other']
                else:
                    return -1
            if type(ans) is int:
                return ans
            cnt += 1

label_num = [0] * 32
total = 0
label_data = [[] for _ in range(32)]
with open('origin_data.json', 'r', encoding='utf-8') as fr:
    line = fr.readline()
    while line:
        data = json.loads(line.strip())
        title = data['title']
        assig = data['assignee']
        abstr = data['abstract']
        ipc = data['ipc']
        label = get_label(title, ipc)
        if label != -1:
            label_num[label] += 1
            total += 1
            label_data[label].append({
                'title': title,
                'assignee': assig,
                'abstract': abstr,
                'label_id': label
            })

        line = fr.readline()

print(f'# Total : {total}')
print('# Each Label:')
for i, num in enumerate(label_num):
    print(f'Label {i:2d} : {num:4d} | {num/total:.1%}')

train_data = []
train_label = [0]*32
test_data = []
test_label = [0]*32
import random
for i, data in enumerate(label_data):
    rate = 0
    if i in [3, 11, 12]:
        rate = 0.05
    elif i in [0, 7, 9, 10, 16, 18, 20, 22, 26]:
        rate = 0.11
    else:
        rate = 0.18
    random.shuffle(data)
    s = int(rate * len(data))
    train_data.extend(data[:s])
    test_data.extend(data[s:])
    train_label[i] += s
    test_label[i] += len(data) - s

random.shuffle(train_data)
random.shuffle(test_data)

train_total = len(train_data)
print(f'\n# Train Total : {train_total}')
print('# Train Each Label:')
for i, num in enumerate(train_label):
    print(f'Label {i:2d} : {num:4d} | {num/train_total:.1%}')
test_total = len(test_data)
print(f'\n# Test Total : {test_total}')
print('# Test Each Label:')
for i, num in enumerate(test_label):
    print(f'Label {i:2d} : {num:4d} | {num/test_total:.1%}')

def save_data(json_file, data):
    with open(json_file, 'w', encoding='utf-8') as fw:
        for d in data:
            json.dump(d, fw, ensure_ascii=False)
            fw.write('\n')
save_data('new_train.json', train_data)
save_data('new_test.json', test_data)

import matplotlib.pyplot as plt
for i in range(32):
    plt.bar(i, train_label[i])
plt.title('Train Label Distribution')
plt.xlabel('Label')
plt.ylabel('Number')
plt.savefig('train_label_distribution.png')

for i in range(32):
    plt.bar(i, test_label[i])
plt.title('Test Label Distribution')
plt.xlabel('Label')
plt.ylabel('Number')
plt.savefig('test_label_distribution.png')