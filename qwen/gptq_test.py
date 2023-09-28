from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTQConfig
from transformers.generation.utils import GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import torch

causal_lm_model_id = "/home/huang/models/opt-350m-gptq-4bit"
tokenizer  = AutoTokenizer.from_pretrained(causal_lm_model_id)
model = AutoModelForCausalLM.from_pretrained(
    causal_lm_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["k_proj", "out_proj", "q_proj", "v_proj", "fc1", "fc2"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

model.print_trainable_parameters()