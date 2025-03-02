# API de Consultas Tuya

Este proyecto implementa una API REST utilizando FastAPI para interactuar con un sistema de recuperación y generación de respuestas (RAG). La API permite realizar consultas sobre una base de conocimiento indexada y generar respuestas contextuales.

## Características

- Consulta a la base de conocimiento y generación de respuestas
- Soporte para índices personalizados (FAISS y Chroma)
- Indexación de documentos en segundo plano
- Verificación del estado de las tareas de indexación
- Interfaz web estática (si se dispone del directorio `static`)

## Instalación

1. Clona el repositorio en tu máquina.
2. Instala los requisitos:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la API:
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

## Uso

Accede a la documentación interactiva en [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints principales

- **POST /api/query**: Realiza una consulta y genera una respuesta.
- **POST /api/index**: Inicia la indexación de documentos.
- **GET /api/index/{index_id}/status**: Obtiene el estado de una tarea de indexación.
- **GET /api/indexes**: Lista todos los índices disponibles.
- **GET /api/health**: Verifica el estado del sistema.
- **GET /**: Mensaje de bienvenida e información sobre la documentación.

## Notas

- Asegúrate de contar con CUDA disponible para el correcto funcionamiento del generador en caso de tener GPU.
- En entornos de producción, adapta la configuración de CORS.
- Actualiza las variables de entorno necesarias antes de iniciar la API.

# Modelos soportados

## Modelos de embeddings

### OpenAI:
- **text-embedding-3-large** (mejor rendimiento)
- **text-embedding-3-small** (más rápido)
- **text-embedding-ada-002** (modelo anterior)

### HuggingFace/Sentence Transformers:
- **all-MiniLM-L6-v2** (predeterminado)
- Otros modelos de Sentence Transformers

## Modelos LLM

### Locales:
- **unsloth/Llama-3.2-1B-Instruct** (predeterminado)
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (más ligero)

### API:
- **gpt-4o** (mejor rendimiento)
- Otros modelos de OpenAI compatibles

# Contenedorización

Se proporciona un **Dockerfile** y un **script de entrada** para facilitar la ejecución en contenedores:

## Construir imagen:
```bash
docker build -t tuya-rag .

## Ejecutar API
```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/faiss_index:/app/faiss_index tuya-rag
