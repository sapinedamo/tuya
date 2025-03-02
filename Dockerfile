# Imagen base con soporte para PyTorch
FROM python:3.9-slim

# Información sobre el mantenedor
LABEL maintainer="Tuya RAG System"
LABEL description="Sistema de Recuperación Aumentada por Generación para Tuya"

# Variables de entorno
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_CACHE_DIR=/app/models

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear y configurar directorio de la aplicación
WORKDIR /app

# Copiar requirements.txt primero para aprovechar caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el código fuente
COPY api/ /app/api/
COPY src/ /app/src/
COPY query_engine.py vectorize_data.py .env ./

# Copiar datos (si están disponibles)
COPY data/ /app/data/
COPY faiss_index/ /app/faiss_index/

# Crear directorio para modelos
RUN mkdir -p /app/models && chmod -R 777 /app/models

# Puerto para la API
EXPOSE 8000

# Volúmenes para persistir datos
VOLUME ["/app/data", "/app/faiss_index", "/app/models"]

# Script de entrada que permite elegir entre API y CLI
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Por defecto, iniciar la API
CMD ["api"]