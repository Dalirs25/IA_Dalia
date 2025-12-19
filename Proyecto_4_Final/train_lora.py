

import os
import argparse
import json
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model


def detect_device_and_dtype():
    """
    Selecciona dispositivo y dtype adecuados para CPU/MPS.
    """
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    # Para entrenamiento en MPS/CPU usar float32 asegura retropropagación estable.
    dtype = torch.float32
    return device, dtype


def suggest_target_modules(model) -> List[str]:
    """
    Intenta sugerir target_modules compatibles con distintas arquitecturas (LLaMA/Mistral/Phi-3).
    Explora nombres de submódulos para decidir.
    """
    module_names = set([name for name, _ in model.named_modules()])
    candidates = [
        # LLaMA/Mistral style
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Phi-3 style (suele tener qkv fusionadas y gate_up)
        ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    ]
    for cand in candidates:
        if any(any(key in m for m in module_names) for key in cand):
            return cand
    # Fallback razonable
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_jsonl_for_check(path: str, n: int = 2):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                json.loads(line)
            except Exception as e:
                raise ValueError(f"JSONL inválido en {path}, línea {i+1}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento LoRA para tutor de algoritmos (CPU/MPS).")
    parser.add_argument("--base", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Modelo base HF")
    parser.add_argument("--train", type=str, required=True, help="Ruta a train.jsonl")
    parser.add_argument("--val", type=str, required=True, help="Ruta a val.jsonl")
    parser.add_argument("--out-dir", type=str, required=True, help="Directorio de salida de los adaptadores LoRA")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Validar datasets
    load_jsonl_for_check(args.train)
    load_jsonl_for_check(args.val)

    device, dtype = detect_device_and_dtype()
    print(f"[INFO] Dispositivo: {device} | dtype: {dtype}")

    print("[INFO] Cargando tokenizer y modelo base...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype
    )
    # Desactivar cache para permitir gradient checkpointing y evitar errores de gradiente
    model.config.use_cache = False
    # Mover a dispositivo
    model.to(torch.device(device))

    # Configurar LoRA
    target_modules = suggest_target_modules(model)
    print(f"[INFO] target_modules LoRA: {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Cargar datasets
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train,
            "validation": args.val
        }
    )

    # Preprocesamiento: tokenizar y crear labels con enmascaramiento de prompt
    def build_prompt(instruction: str) -> str:
        return f"Instrucción: {instruction}\nRespuesta: "

    eos = tokenizer.eos_token or "</s>"

    def preprocess(example: Dict):
        instr = example.get("instruction", "").strip()
        resp = example.get("response", "").strip()
        prompt_text = build_prompt(instr)
        full_text = prompt_text + resp + eos

        # Tokenizar por separado para poder enmascarar el prompt
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=args.max_seq_len, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, truncation=True, max_length=args.max_seq_len, add_special_tokens=False)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels
        }

    tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

    # Data collator estándar (no MLM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("[INFO] Configurando entrenamiento...")
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        seed=args.seed,
        dataloader_num_workers=0,  # seguro para macOS
        dataloader_pin_memory=False,  # evitar warning en MPS
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,  # simplificar en MPS
        optim="adamw_torch",  # evitar fused optimizer en CPU/MPS
        report_to=[],
        use_mps_device=True if device == "mps" else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    print("[INFO] Iniciando entrenamiento...")
    trainer.train()

    print("[INFO] Guardando adaptadores LoRA en", args.out_dir)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("[DONE] Entrenamiento completado.")
    

if __name__ == "__main__":
    main()
