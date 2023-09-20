import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

modelpath = "/mnt/d/models/Baichuan2-7B-Chat"
quant8_saved_dir = "/mnt/d/models/Baichuan2-7B-Chat-8bits"

# 保存 8bits量化模型
# model = AutoModelForCausalLM.from_pretrained(modelpath, load_in_8bit=True, device_map="cpu", trust_remote_code=True)
# model.save_pretrained(quant8_saved_dir)


tokenizer = AutoTokenizer.from_pretrained(quant8_saved_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(quant8_saved_dir, device_map="auto", trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(quant8_saved_dir)
model = model.eval()

messages = []
start_time = time.time()
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
end_time = time.time()
run_time = end_time - start_time
print("Baichuan2: ", response)
print(f"running time: {run_time:.4f} s")

# Baichuan2:  "温故而知新"是一个汉语成语，出自《论语·为政》。这个成语的意思是通过回顾过去的事情，可以发现
# 新的知识和道理。它强调了学习和成长的过程应该包括两个方面：一方面是要不断复习和巩固过去的知识，另一方面是要
# 通过实践和思考来获得新的见解和收获。

# 这个成语鼓励我们要有批判性思维，不仅要接受现有的知识，还要不断地质疑和探索。通过对比新旧知识，我们可以
# 发现其中的联系和差异，从而更好地理解和应用新知识。同时，这个成语也提醒我们要善于总结和反思，从过去的经验中
# 汲取教训，为未来的发展做好准备。
# running time: 16.7600 s