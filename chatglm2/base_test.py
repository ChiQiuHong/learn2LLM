import time

from transformers import AutoTokenizer, AutoModel

modelpath = "/home/huang/models/chatglm2-6b"

tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
model = AutoModel.from_pretrained(modelpath, trust_remote_code=True).quantize(8).cuda()
model = model.eval()

start_time = time.time()
print("USER: 你好")
response, history = model.chat(tokenizer, "你好", history=[])
end_time = time.time()
run_time = end_time - start_time
print("ChatGLM2: ", response)
print(f"running time: {run_time:.4f} s")

# start_time = time.time()
# print("USER: 晚上睡不着应该怎么办")
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# end_time = time.time()
# run_time = end_time - start_time
# print("ChatGLM2: ", response)
# print(f"running time: {run_time:.4f} s")