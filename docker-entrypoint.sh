#!/bin/bash
set -e

# Verificar si los índices vectoriales existen
if [ ! -f "/app/faiss_index/index.faiss" ]; then
    echo "⚠️ No se encontró el índice FAISS. Verificar que el volumen esté montado correctamente."
fi

# Verificar argumentos de entrada
if [ "$1" = "api" ]; then
    echo "🚀 Iniciando API en puerto 8000..."
    exec python -m api.app
elif [ "$1" = "cli" ]; then
    echo "🖥️ Iniciando modo CLI interactivo..."
    shift
    exec python query_engine.py --interactive "$@"
elif [ "$1" = "vectorize" ]; then
    echo "🔄 Procesando documentos y creando índice vectorial..."
    shift
    exec python vectorize_data.py "$@"
elif [ "$1" = "shell" ]; then
    echo "💻 Iniciando shell..."
    exec /bin/bash
else
    # Si los argumentos no corresponden a ningún comando conocido, asumimos que son para query_engine.py
    echo "⚙️ Ejecutando comando personalizado..."
    exec python query_engine.py "$@"
fi