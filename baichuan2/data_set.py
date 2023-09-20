import torch
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Seq2SeqDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):
        """
        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_len: 模型输入最大长度
            max_src_len: 模型源文本最大长度
            prompt_text: 提示文本
        """
        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []
        # 遍历数据文件内容
        with open(data_path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh):
                # 利用json.loads进行数据加载
                sample = json.loads(line.strip())
                # 利用分词器，对源文本和提示文本进行分词
                
                src_tokens = tokenizer.tokenize(sample['text'])

                prompt_tokens = tokenizer.tokenize(prompt_text)
                print("src_tokens:", src_tokens)
                print("prompt tokens: ", prompt_tokens)
                # 对分词后的源文本进行截断
                if len(src_tokens) > max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]
                # 对分词后的目标文本进行截断
                tgt_tokens = tokenizer.tokenize(sample['answer'])
                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                print("tgt tokens", tgt_tokens)
                # 将分词后的源文本和目标文本进行拼接，并转换成模型所需的索引ID格式
                tokens = prompt_tokens + src_tokens + ['[gMASK]', '<sop>'] + tgt_tokens + ['<eop>']
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                print("input_ids", input_ids)
                # 对于训练模型的标签，仅仅保留目标文本索引ID，其他内容设置成-100，模型不计算对应的损失
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]
                # 对最终结果进行填充，填充到模型的最大长度
                ped_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * ped_len
                labels = labels + [-100] * ped_len
                # 将每个样本进行保存，用于后续训练使用
                self.all_data.append({
                    'text': sample['text'],
                    'answer': sample['answer'],
                    'input_ids': input_ids,
                    'labels': labels
                })

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        return self.all_data[index]


class Seq2SeqDataSet2(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):
        """
        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_len: 模型输入最大长度
            max_src_len: 模型源文本最大长度
            prompt_text: 提示文本
        """
        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []
        # 遍历数据文件内容
        with open(data_path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh):
                # 利用json.loads进行数据加载
                sample = json.loads(line.strip())
                # 利用分词器，对源文本和提示文本进行分词
                
                src_ids = tokenizer.encode(text= prompt_text + sample['text'],
                                           add_special_tokens=True,
                                           truncation=True,
                                           max_length=max_src_len)
                
                tgt_ids = tokenizer.encode(text= sample['answer'],
                                           add_special_tokens=False,
                                           truncation=True,
                                           max_length=max_tgt_len)

                input_ids = src_ids + tgt_ids + [tokenizer.eos_token_id]

                # 对于训练模型的标签，仅仅保留目标文本索引ID，其他内容设置成-100，模型不计算对应的损失
                labels = [-100] * len(src_ids) + tgt_ids + [tokenizer.eos_token_id]

                # 对最终结果进行填充，填充到模型的最大长度
                ped_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * ped_len
                labels = labels + [-100] * ped_len
                # 将每个样本进行保存，用于后续训练使用
                self.all_data.append({
                    'text': sample['text'],
                    'answer': sample['answer'],
                    'input_ids': input_ids,
                    'labels': labels
                })

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        return self.all_data[index]


def coll_fn(batch):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, labels_list = [], []
    for instance in batch:
        # 将input_ids和token_type_ids添加到对应的list中
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=20003),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=20003)}
