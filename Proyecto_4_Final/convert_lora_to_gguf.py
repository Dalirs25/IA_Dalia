
"""
Convierte un LoRA (PEFT) entrenado a un modelo GGUF fusionado para usar en Ollama.
- Fusiona adaptadores LoRA con el modelo base (merge_and_unload).
- Guarda el modelo HF fusionado en un dir temporal.
- Ejecuta llama.cpp/convert-hf-to-gguf.py para producir un .gguf (f16 por defecto).
- No requiere NVIDIA. Funciona en CPU/Mac MPS.

Requisitos:
  pip install transformers peft torch sentencepiece safetensors

Ejemplo:
  python projects/fine_tuning_project/export/convert_lora_to_gguf.py \
    --base microsoft/Phi-3-mini-4k-instruct \
    --lora projects/fine_tuning_project/finetune/lora/outputs/tutor-lora \
    --out projects/fine_tuning_project/export/tutor_gguf/tutor_f16.gguf \
    --dtype float16

Notas:
- Si llama.cpp no está presente, se clonará en projects/fine_tuning_project/export/llama.cpp (git requerido).
- Para algunos modelos puede ser necesario instalar adicionalmente sentencepiece.
"""

import argparse
import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def ensure_llama_cpp_exists(llama_cpp_dir: Path):
    if llama_cpp_dir.exists():
        return
    print(f"[INFO] Clonando llama.cpp en {llama_cpp_dir} ...")
    url = "https://github.com/ggerganov/llama.cpp.git"
    subprocess.run(["git", "clone", "--depth", "1", url, str(llama_cpp_dir)], check=True)
    print("[INFO] llama.cpp clonado.")


def main():
    ap = argparse.ArgumentParser(description="Fusiona LoRA + base y convierte a GGUF para Ollama.")
    ap.add_argument("--base", required=True, help="Modelo base HF (nombre o ruta local)")
    ap.add_argument("--lora", required=True, help="Ruta del LoRA entrenado (directorio con adapter_config.json)")
    ap.add_argument("--out", required=True, help="Ruta de salida del GGUF (ej. export/tutor_gguf/tutor_f16.gguf)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Precisión del modelo fusionado antes de GGUF")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cargar tokenizer y modelo base
    print("🔹 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    print("🔹 Cargando modelo base...")
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch_dtype)

    # Cargar LoRA
    print("🔹 Cargando LoRA...")
    model = PeftModel.from_pretrained(base_model, args.lora)

    # Fusionar
    print("🔹 Fusionando LoRA con el modelo base...")
    model = model.merge_and_unload()

    # Guardar temporalmente el modelo fusionado en formato HF
    tmp_dir = Path(tempfile.mkdtemp(prefix="merged_hf_"))
    print(f"🔹 Guardando modelo fusionado en {tmp_dir} ...")
    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)

    # Asegurar llama.cpp presente
    script_dir = Path(__file__).resolve().parent
    llama_cpp_dir = script_dir / "llama.cpp"
    try:
        ensure_llama_cpp_exists(llama_cpp_dir)
    except Exception as e:
        print(f"[ERROR] No se pudo clonar llama.cpp automáticamente: {e}")
        print("Clona manualmente: git clone https://github.com/ggerganov/llama.cpp.git projects/fine_tuning_project/export/llama.cpp")
        sys.exit(1)

    # Localizar script de conversión (nombres cambiaron en versiones recientes de llama.cpp)
    candidates = [
        llama_cpp_dir / "convert-hf-to-gguf.py",
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "scripts" / "convert-hf-to-gguf.py",
        llama_cpp_dir / "scripts" / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert_hf_to_gguf_update.py",
        llama_cpp_dir / "scripts" / "convert_hf_to_gguf_update.py",
    ]
    convert_script = None
    for c in candidates:
        if c.exists():
            convert_script = c
            break
    if convert_script is None:
        print(f"[ERROR] No se encontró el script de conversión (convert*_to_gguf.py) en {llama_cpp_dir}")
        sys.exit(1)

    # Tipo de salida
    outtype = "f16" if args.dtype == "float16" else "f32"

    print("🔹 Convirtiendo a GGUF...")
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(tmp_dir),
            "--outfile",
            str(out_path),
            "--outtype",
            outtype,
        ],
        check=True,
    )

    # Limpiar temporal
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    print(f"✅ Conversión completa: {out_path}")


if __name__ == "__main__":
    main()
