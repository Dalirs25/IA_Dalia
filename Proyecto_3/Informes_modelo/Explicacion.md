# Explicación del Proyecto: 

## 1. Introducción

El objetivo principal fue aplicar conceptos vistos en clase, como modelos de lenguaje, embeddings y recuperación de información, en un caso práctico que no fuera únicamente técnico, sino también analítico y reflexivo.

El tema elegido fue **“La Generación Z y la crisis de sentido”**, ya que es un fenómeno actual que se manifiesta claramente en redes sociales y plataformas digitales. A través de este proyecto busqué analizar cómo la tecnología, los algoritmos y la hiperconectividad influyen en la identidad, emociones y percepciones de esta generación.


## 2. Organización general del proyecto

El proyecto está organizado en diferentes carpetas y archivos, cada uno con una función específica:

* **BD_vector/**: contiene la base de datos vectorial generada con Chroma, donde se almacenan los embeddings de los textos.
* **datasets/**: incluye los archivos CSV utilizados como fuente de datos (redes sociales, artículos y comentarios de YouTube).
* **Informes_modelo/**: carpeta donde se guardan los informes finales generados automáticamente en formato Markdown (.md).
* **Scripts en Python**: archivos encargados del scraping, vectorización y generación de respuestas.
* **requirements.txt**: lista de librerías necesarias para ejecutar el proyecto.


## 3. Fuentes de datos utilizadas

Para el análisis utilicé diferentes tipos de datos, con el fin de tener una visión más amplia y no depender de una sola fuente:


El archivo **dataset_sintetico_5000_ampliado.csv** contiene textos de publicaciones en redes sociales. 

Este dataset fue útil para analizar emociones, patrones de lenguaje y temas recurrentes relacionados con presión digital, burnout y sentido de vida.


El archivo **articles_chunks.csv** contiene fragmentos de artículos periodísticos y de divulgación. Estos textos aportan un contexto más informativo y teórico, complementando las opiniones de redes sociales.


Los comentarios se obtuvieron mediante un script de scraping (**youtube_comments_scraper.py**). Estos comentarios reflejan opiniones más espontáneas y emocionales, lo que resulta valioso para identificar percepciones reales de los usuarios.


## 4. Proceso de vectorización y base de datos

Uno de los pasos más importantes del proyecto fue la creación de la base de datos vectorial, que se realiza en el archivo **vectorizador.py**.

En este script:

* Se cargan los archivos CSV.
* Cada texto se transforma en un embedding utilizando un modelo de embeddings.
* Los embeddings se almacenan en una base de datos Chroma.

El objetivo de este paso es que el sistema pueda buscar fragmentos de texto relevantes según una pregunta, no solo por coincidencia de palabras, sino por similitud semántica.


## 5. Recuperación aumentada por generación (RAG)

El corazón del proyecto es el enfoque **RAG (Retrieval-Augmented Generation)**. Este enfoque combina dos procesos:

1. **Recuperación**: cuando se hace una pregunta, el sistema busca en la base vectorial los textos más relacionados.
2. **Generación**: esos textos recuperados se pasan como contexto a un modelo de lenguaje, que genera una respuesta fundamentada.

Este enfoque se implementa principalmente en el archivo **ollama_rag.py**.


## 6. Archivo ollama_rag.py

Este archivo se encarga de:

* Leer las preguntas desde un archivo de texto.
* Recuperar información relevante desde la base vectorial.
* Enviar el contexto y la pregunta a un modelo de lenguaje (Llama 3.2).
* Generar respuestas extensas y estructuradas.
* Guardar los resultados en un archivo Markdown.


## 7. Preguntas de investigación

Las preguntas planteadas buscan explorar diferentes dimensiones de la crisis de sentido en la Generación Z, por ejemplo:

* Vacío existencial en redes sociales.
* Influencia de los algoritmos en la identidad.
* Emociones asociadas al burnout digital.
* Autonomía, libertad y control algorítmico.
* Identidad líquida y yo digital.


## 8. Resultados y generación del informe

El resultado final del proyecto es un **informe en formato Markdown**

Este informe se genera automáticamente, pero el valor del proyecto está en el diseño del proceso y en la selección del enfoque de análisis.


## 9. Aprendizajes personales

A lo largo de este proyecto me di cuenta de que la inteligencia artificial no solo sirve para programar o resolver problemas técnicos, sino que también puede utilizarse para analizar temas sociales y humanos. Durante el desarrollo entendí mejor cómo funcionan los modelos de lenguaje, los embeddings y las bases de datos vectoriales, y cómo estas herramientas pueden ayudar a reflexionar sobre situaciones reales. Además, este trabajo me permitió combinar la parte técnica con el pensamiento crítico, ya que no solo consistió en ejecutar código, sino en analizar información y sacar conclusiones sobre un tema complejo como la crisis de sentido en la Generación Z