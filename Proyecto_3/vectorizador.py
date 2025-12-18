from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import json

# Configuración del modelo de embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "C:/python_projects/IA_Dalia/Proyecto_3/BD_vector"

# Función auxiliar para cargar CSVs
def load_csv_data(filepath, text_column, metadata_columns, source_name):
    docs = []
    if os.path.exists(filepath):
        print(f"Cargando {source_name} desde CSV...")
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            # Construir metadatos dinámicamente
            meta = {"source": source_name}
            for col in metadata_columns:
                if col in row:
                    meta[col] = str(row[col])
            
            # Crear documento
            if text_column in row and pd.notna(row[text_column]):
                doc = Document(
                    page_content=str(row[text_column]),
                    metadata=meta
                )
                docs.append(doc)
    else:
        print(f"Advertencia: No se encontró {filepath}")
    return docs

# Función auxiliar para cargar JSONL
def load_jsonl_data(filepath, text_column, metadata_columns, source_name):
    docs = []
    if os.path.exists(filepath):
        print(f"Cargando {source_name} desde JSONL...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    meta = {"source": source_name}
                    for col in metadata_columns:
                        if col in data:
                            meta[col] = str(data[col])
                    
                    if text_column in data and data[text_column]:
                        doc = Document(
                            page_content=str(data[text_column]),
                            metadata=meta
                        )
                        docs.append(doc)
                except json.JSONDecodeError:
                    continue
    else:
        print(f"Advertencia: No se encontró {filepath}")
    return docs



# Verificar si la base de datos ya existe
if not os.path.exists(db_location):
    print("Creando nueva base de datos vectorial...")
    all_documents = []

    
    all_documents.extend(load_csv_data(
        "C:/python_projects/IA_Dalia/Proyecto_3/dataset_sintetico_5000_ampliado.csv", 
        text_column="texto", 
        metadata_columns=["tema", "sentimiento", "fecha"], 
        source_name="redes_sociales_sintetico"
    ))
    all_documents.extend(load_csv_data(
        "C:/python_projects/IA_Dalia/Proyecto_3/articles_chunks.csv", 
        text_column="text", 
        metadata_columns=["title", "url", "author"], 
        source_name="articulos_externos"
    ))
    all_documents.extend(load_csv_data(
        "C:/python_projects/IA_Dalia/Proyecto_3/youtube_comments_vm5tGIDUS9E.csv", 
        text_column="text", 
        metadata_columns=["author", "video_id", "like_count"], 
        source_name="youtube_comentarios"
    ))

    print(f"Total de documentos procesados: {len(all_documents)}")

   
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=db_location,
        collection_name="genz_research"
    )
    print("Base de datos creada exitosamente.")

else:
    print("La base de datos ya existe. Conectando...")
    vector_store = Chroma(
        persist_directory=db_location,
        embedding_function=embeddings,
        collection_name="genz_research"
    )

# Exponer el retriever para que lo use el otro script
retriever = vector_store.as_retriever(
    search_type="mmr", # "mmr" busca diversidad en las respuestas, no solo similitud
    search_kwargs={"k": 10} # Recuperar 10 fragmentos para tener contexto suficiente
)