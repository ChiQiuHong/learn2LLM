# learn2LLM

## 信息抽取实战

信息抽取数据集采自CCF2022工业知识图谱关系抽取比赛，并针对一个工业制造领域的相关故障文本，抽取4种类型的实体（部件单元、性能表征、故障状态和检测工具）以及4种类型的关系（部件故障、性能故障、检测工具、组成）。由于数据量较少，在实战过程中，随机抽取50条数据作为测试数据，其余数据作为训练数据集。

| Model | PEFT Method | F1 Score | max_len | max_src_len | BatchSize | Epoch | GPU Usage |
| :---: | :---------: | :------: | :-----: | :---------: | :-------: | :---: | :-------: |
| Baichuan2-7B-Chat | LoRA(r=8) | 0.4785 | 1024 | - | 2 | 5 | 19.3GB |
| Baichuan2-13B-Chat | QLoRA(r=64) | 0.5723 | 768 | - | 2 | 5 | 20.8GB |