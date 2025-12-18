
import os
from datetime import datetime
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorizador import retriever
import time

# --- CONFIGURACIÓN ---
model = OllamaLLM(model="llama3.2:latest") 
INPUT_FILE = "C:/python_projects/IA_Dalia/Proyecto_3/preguntas.txt"
OUTPUT_FILE = "C:/python_projects/IA_Dalia/Proyecto_3/Informes_modelo/INFORME_FINAL_COMPLETO_2.md"

# --- TEMPLATE DEL PROMPT (Científico/Filosófico) ---
template = """
Eres un investigador experto en filosofía y análisis de datos. Proyecto: "La Generación Z y la Crisis de Sentido".

OBJETIVO: Responde la siguiente pregunta de investigación sintetizando:
1. TEORÍA: Conceptos filosóficos (Heidegger, Han, Bauman, etc.) presentes en el contexto.
2. EVIDENCIA: Datos empíricos (YouTube, encuestas, Reddit) presentes en el contexto.

CONTEXTO RECUPERADO:
{context}

PREGUNTA DE INVESTIGACIÓN: 
{question}

INSTRUCCIONES DE RESPUESTA:
- Escribe un análisis profundo y estructurado (mínimo 2 párrafos).
- Cita las fuentes teóricas y empíricas explícitamente.
- Si hay contradicciones entre la teoría y los datos, señálalas.
- Responde en español académico.

ANÁLISIS:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def cargar_preguntas(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: No se encontró el archivo '{filepath}'")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        # Lee las líneas y quita los espacios vacíos
        return [line.strip() for line in f if line.strip()]

def procesar_cuestionario():
    print("\n=== INICIANDO ANÁLISIS AUTOMATIZADO DE PROYECTO DE IA ===")
    
    # 1. Cargar preguntas
    preguntas = cargar_preguntas(INPUT_FILE)
    if not preguntas:
        return

    total = len(preguntas)
    print(f"📂 Se encontraron {total} preguntas en '{INPUT_FILE}'.")
    print(f"📝 El resultado se escribirá en '{OUTPUT_FILE}'\n")

    # 2. Preparar el archivo de salida (Escribir cabecera)
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Informe de Investigación: Crisis de Sentido Gen Z\n")
        f.write(f"**Fecha de generación:** {timestamp}\n")
        f.write(f"**Modelo:** Deepseek-r1 + RAG\n")
        f.write(f"**Total de preguntas:** {total}\n")
        f.write("---\n\n")

    # 3. Bucle de procesamiento
    for i, question in enumerate(preguntas, 1):
        print(f"⏳ Procesando pregunta {i}/{total}: {question[:50]}...")
        start_time = time.time()

        # --- A. RECUPERACIÓN (RAG) ---
        docs = retriever.invoke(question)
        
        # Formatear contexto enriquecido con fuentes
        context_text = ""
        sources_used = set() # Para listar fuentes al final de la respuesta
        
        for doc in docs:
            source_type = doc.metadata.get("source", "desconocido").upper()
            content = doc.page_content.replace("\n", " ")
            context_text += f"[{source_type}]: {content}\n\n"
            sources_used.add(source_type)

        # --- B. GENERACIÓN (LLM) ---
        response = chain.invoke({"context": context_text, "question": question})

        # --- C. ESCRITURA EN EL ARCHIVO ---
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"## {i}. {question}\n\n")
            f.write(f"{response}\n\n")
            
            # Sección de Fuentes (Metadatos)
            f.write("**Fuentes consultadas para este análisis:**\n")
            for src in sources_used:
                f.write(f"- *{src}*\n")
            f.write("\n---\n\n") # Separador
        
        elapsed = time.time() - start_time
        print(f"✅ Terminada en {elapsed:.2f}s.\n")

    print(f"\n🎉 ¡PROCESO COMPLETADO! Revisa el archivo: {OUTPUT_FILE}")

if __name__ == "__main__":
    procesar_cuestionario()