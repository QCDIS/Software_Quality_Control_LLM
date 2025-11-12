#!/usr/bin/env python
# coding: utf-8



import torch, platform, os
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("Total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
else:
    print("Running on CPU")




from dataclasses import dataclass

# You can change these to your preferred model/dataset
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"   # Small, permissive chat model
DATASET_NAME = "HuggingFaceTB/everyday-conversations-llama3.1-2k"             

# Training settings (tuned for small GPUs; adjust as needed)
MAX_SEQ_LENGTH = 512          # reduce to 256 if you run OOM
PER_DEVICE_BATCH = 1          # micro-batch size
GRAD_ACCUM_STEPS = 8          # to keep an effective batch size
NUM_EPOCHS = 1                # demo; increase to 2–3 for better results
LEARNING_RATE = 2e-4
OUTPUT_DIR = "llama-sft"

USE_4BIT = False               # 4-bit quantization via bitsandbytes
USE_BF16 = True               # set False and use fp16=True if your GPU prefers fp16 (e.g., T4/V100)
SEED = 42




from datasets import load_dataset

# The dataset has fields like: 'instruction', 'input', 'output'
raw_ds = load_dataset(DATASET_NAME)
print(raw_ds)
print(raw_ds['train_sft'][3])




SYSTEM_PROMPT = "You are a helpful, honest, and concise assistant."
def format_example(example):
    instr = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()

    if inp:
        user = f"{instr}\n\nInput:\n{inp}"
    else:
        user = instr

    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user}\n<|assistant|>\n{out}"
    return {"text": prompt}

# Map to a single 'text' field used for SFT
processed = raw_ds.map(format_example, remove_columns=raw_ds["train_sft"].column_names)
print(processed)
print(processed["train_sft"][0]["text"][:400])




from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
# Ensure pad_token exists (some LLMs don't have it set by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
)

tokenizer.model_max_length = MAX_SEQ_LENGTH
tokenizer.padding_side = "right"




from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=USE_BF16,            # set fp16=True if needed instead
    fp16=(not USE_BF16),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=MAX_SEQ_LENGTH,   # TRL >= 0.20
    packing=False,                   # keep False for stability
    seed=SEED,
    report_to="none",
)




trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=processed["train_sft"],
    args=sft_config,
    peft_config=peft_config,
    #dataset_text_field="text",  # the field produced by our formatter
)




trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=processed["train_sft"],
    args=sft_config,
    peft_config=peft_config,
    #dataset_text_field="text",  # the field produced by our formatter
)




trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved to:", OUTPUT_DIR)




from transformers import TextStreamer

prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nExplain LoRA fine-tuning in one paragraph.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer
    )




with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer
    )

