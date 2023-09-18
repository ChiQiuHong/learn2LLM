import json
import random


def ftdata_process(ori_path, train_path, test_path):
    data = []
    with open(ori_path, 'r', encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            sample = json.loads(line.strip())
            # 从原始文件中抽取三元组内容，三元组内容之间用“_”连接
            spo_list = []
            for spo in sample['spo_list']:
                spo_list.append('_'.join([spo['h']["name"], spo['relation'], spo['t']['name']]))
            # 多个三元组之间用“\n”连接
            data.append({"text": sample["text"], "answer": "\n".join(spo_list)})
        # 随机打乱数据集
        random.shuffle(data)

        fin_0 = open(train_path, "w", encoding="utf-8")
        fin_1 = open(test_path, "w", encoding="utf-8")
        for i, sample in enumerate(data):
            # 随机抽取50条数据作为测试集
            if i < 50:
                fin_1.write(json.dumps(sample, ensure_ascii=False) + '\n')
            # 其余作为训练集
            else:
                fin_0.write(json.dumps(sample, ensure_ascii=False) + '\n')
        fin_0.close()
        fin_1.close()


if __name__ == '__main__':
    ori_path = "dataset/fine-tuning/ori_data.json"
    train_path = "dataset/fine-tuning/spo_0.json"
    test_path = "dataset/fine-tuning/spo_1.json"
    ftdata_process(ori_path, train_path, test_path)