# Proyecto: Fine‑Tuning con LoRA para Tutor de Algoritmos

## 1. Introducción

El objetivo principal fue entrenar un modelo base para convertirlo en un **tutor de algoritmos**, capaz de explicar conceptos paso a paso, analizar complejidad temporal y espacial, y señalar errores comunes. Más que solo entrenar un modelo, el enfoque del proyecto fue **entender todo el flujo completo**: preparación de datos, entrenamiento con LoRA, guardado de adaptadores, conversión a GGUF y despliegue final en Ollama.

## 2. Organización general del proyecto

El proyecto está organizado de forma modular para separar claramente cada etapa del proceso:

* **processed/**: contiene los archivos `train.jsonl`, `val.jsonl`, `test.jsonl` y `ollama_dataset.jsonl`, que representan los datasets ya procesados y listos para entrenamiento o evaluación.
* **tutor-lora5epoch/**: carpeta donde se guardan los resultados del entrenamiento LoRA, incluyendo los checkpoints y los adaptadores finales.
* **exportados/**: contiene el modelo final en formato GGUF (`tutor_5epoch.gguf`) listo para ser usado en Ollama.
* **train_lora.py**: script principal de entrenamiento.
* **convert_lora_to_gguf.py**: script para fusionar el modelo base con los adaptadores LoRA y exportarlo a GGUF.
* **Modelfile.lora**: archivo de configuración para crear el modelo final en Ollama.
* **requirements.txt**: dependencias necesarias para ejecutar todo el proyecto.

## 3. Dataset y preparación de datos

El dataset está compuesto por instrucciones y respuestas enfocadas en **algoritmos y estructuras de datos**, con distintos niveles de dificultad (básico, intermedio y avanzado). Los ejemplos incluyen:

* Explicaciones teóricas de algoritmos (ordenamiento, grafos, programación dinámica, backtracking).
* Análisis de complejidad temporal y espacial.
* Corrección de errores comunes en fragmentos de código.
* Ejercicios resueltos paso a paso.

Los datos se almacenaron en formato **JSONL**, separando entrenamiento (`train.jsonl`), validación (`val.jsonl`) y pruebas (`test.jsonl`). Este formato facilita el streaming de datos y es compatible directamente con la librería `datasets` de Hugging Face.

Durante el preprocesamiento, el prompt se construye con una estructura clara:

```
Instrucción: <texto>
Respuesta:
```

El prompt se enmascara en las etiquetas para que el modelo aprenda únicamente a generar la respuesta, evitando que “memorice” la instrucción.


## 4. Entrenamiento con LoRA (train_lora.py)

El entrenamiento se realizó utilizando un **modelo base de Hugging Face**. Esto fue una decisión consciente para demostrar que el fine‑tuning con LoRA es viable incluso sin GPU.

En el script `train_lora.py`:

* Se detecta automáticamente el dispositivo disponible (CPU o MPS).
* Se cargan el tokenizer y el modelo base.
* Se desactiva el `use_cache` para evitar problemas durante la retropropagación.
* Se configuran los parámetros LoRA (r, alpha y dropout).
* Se seleccionan dinámicamente los `target_modules` compatibles con la arquitectura del modelo.

LoRA permite entrenar solo una pequeña parte del modelo (adaptadores), lo que reduce drásticamente el consumo de memoria y tiempo, manteniendo el conocimiento general del modelo base.

El entrenamiento se realizó por **5 épocas**, guardando checkpoints intermedios y finalmente los adaptadores entrenados.

## 5. Resultados del entrenamiento

Como resultado del entrenamiento se generó una carpeta que contiene:

* `adapter_model.safetensors`: pesos entrenados de LoRA.
* `adapter_config.json`: configuración del entrenamiento.
* `tokenizer.json` y archivos relacionados.
* Checkpoints intermedios para control del proceso.

Estos archivos no representan un modelo completo, sino **adaptadores que modifican el comportamiento del modelo base**.

## 6. Conversión a GGUF

Para poder usar el modelo en **Ollama**, fue necesario convertir el modelo base junto con los adaptadores LoRA a un solo archivo GGUF. Este proceso se realizó con el script `convert_lora_to_gguf.py`.

Este paso fusiona:

* El modelo base.
* Los pesos LoRA entrenados.

El resultado final es el archivo:

```
exportados/tutor_5epoch.gguf
```

## 7. Integración con Ollama (Modelfile.lora)

Para desplegar el modelo se creó un `Modelfile.lora`, donde se especifica:

* El archivo GGUF a utilizar.
* El mensaje de sistema que define el comportamiento del modelo.

El modelo fue configurado para actuar como un **tutor paciente, claro y didáctico**, que explica paso a paso, incluye ejemplos y analiza complejidad, respondiendo siempre en español.


## 8. Aprendizajes personales

Este proyecto me ayudó a entender realmente cómo funciona el fine‑tuning con LoRA más allá de la teoría. Aprendí que no es necesario reentrenar un modelo completo para especializarlo, sino que se pueden entrenar adaptadores pequeños de forma eficiente. También comprendí mejor el flujo completo de un modelo de lenguaje: desde los datos, el entrenamiento, la exportación y finalmente su uso en una aplicación real.

En conclusión, este proyecto demuestra una aplicación práctica y completa del fine‑tuning con LoRA para crear un tutor de algoritmos especializado. A través de este trabajo pude integrar conocimientos de modelos de lenguaje, entrenamiento eficiente y despliegue de modelos, logrando un sistema funcional.
