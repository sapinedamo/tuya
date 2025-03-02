"""
Script para procesar documentos extraídos de web scraping y vectorizarlos.

Este módulo toma documentos en formato JSON, los divide en fragmentos pequeños
y los convierte en vectores para su uso en un sistema de recuperación vectorial.
"""

import json
import os
import logging
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vectorize_data")

# Constantes
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class ProcessingConfig:
    """Configuración para el procesamiento de documentos."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    store_type: str = "faiss"
    openai_api_key: Optional[str] = None


def load_documents_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Carga los documentos desde un archivo JSON.
    
    Args:
        json_path: Ruta al archivo JSON que contiene los documentos.
        
    Returns:
        Lista de documentos cargados.
        
    Raises:
        FileNotFoundError: Si el archivo JSON no existe.
        json.JSONDecodeError: Si el archivo JSON está mal formateado.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            documents = json.load(file)
        return documents
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo: {json_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error al decodificar JSON en: {json_path}")
        raise


def process_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Procesa los documentos para preparar el chunking.
    
    Args:
        documents: Lista de documentos con contenido y metadatos.
        
    Returns:
        Lista de documentos procesados con texto completo y metadatos.
    """
    processed_docs = []
    
    for doc in documents:
        # Extraer contenido relevante
        content = doc.get('content', '')
        url = doc.get('url', '')
        title = doc.get('title', '')
        
        if not content:
            logger.warning(f"Documento sin contenido: {url or title}")
            continue
            
        # Agregar título al inicio del contenido para mejorar contexto
        full_text = f"{title}\n\n{content}" if title else content
        
        # Metadatos que se conservarán con cada chunk
        metadata = {
            "source": url,
            "title": title
        }
        
        processed_docs.append({"text": full_text, "metadata": metadata})
    
    return processed_docs


def split_documents(processed_docs: List[Dict[str, Any]], 
                   config: ProcessingConfig) -> List[Dict[str, Any]]:
    """
    Divide los documentos en chunks más pequeños.
    
    Args:
        processed_docs: Lista de documentos procesados.
        config: Configuración para el chunking.
        
    Returns:
        Lista de chunks con texto y metadatos.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    for doc in processed_docs:
        doc_chunks = text_splitter.split_text(doc["text"])
        for chunk in doc_chunks:
            chunks.append({"text": chunk, "metadata": doc["metadata"]})
    
    return chunks


def create_vector_store(chunks: List[Dict[str, Any]], config: ProcessingConfig, output_dir: Optional[str] = None):
    """
    Crea un vector store con los chunks vectorizados.
    
    Args:
        chunks: Lista de chunks a vectorizar.
        config: Configuración para el embedding y el vector store.
        output_dir: Directorio donde guardar el vector store (opcional).
        
    Returns:
        Vector store creado.
        
    Raises:
        ValueError: Si el tipo de vector store no es soportado.
    """
    # Inicializar el modelo de embeddings
    try:
        if config.embedding_model.startswith("text-embedding-"):
            # Usar OpenAI embeddings con API key explícita
            from langchain_openai import OpenAIEmbeddings
            
            if config.openai_api_key:
                # Usar la API key proporcionada
                embeddings = OpenAIEmbeddings(
                    model=config.embedding_model,
                    openai_api_key=config.openai_api_key
                )
            else:
                # Usar la variable de entorno OPENAI_API_KEY
                embeddings = OpenAIEmbeddings(model=config.embedding_model)
                
            logger.info(f"Usando embeddings de OpenAI: {config.embedding_model}")
        else:
            # Usar HuggingFace embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=f"sentence-transformers/{config.embedding_model}"
            )
            logger.info(f"Usando embeddings de HuggingFace: {config.embedding_model}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo de embeddings: {e}")
        raise
    
    # Preparar textos y metadatos para el vector store
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Crear vector store según el tipo elegido
    if config.store_type.lower() == "faiss":
        try:
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas
            )
            
            # Guardar el vector store
            if output_dir is None:
                output_dir = "faiss_index"
                
            os.makedirs(output_dir, exist_ok=True)
            vector_store.save_local(output_dir)
            logger.info(f"Vector store FAISS guardado en {output_dir}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error al crear el vector store FAISS: {e}")
            raise
            
    elif config.store_type.lower() == "chroma":
        try:
            # Configurar directorio para Chroma
            if output_dir is None:
                output_dir = "chroma_db"
                
            os.makedirs(output_dir, exist_ok=True)
            
            vector_store = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                persist_directory=output_dir
            )
            
            # Persistir vector store
            vector_store.persist()
            logger.info(f"Vector store Chroma guardado en {output_dir}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error al crear el vector store Chroma: {e}")
            raise
            
    else:
        raise ValueError(f"Tipo de vector store no soportado: {config.store_type}")


def perform_test_query(vector_store, query: str = "Cómo pago mi tarjeta Tuya", k: int = 3):
    """Realiza una búsqueda de prueba en el vector store."""
    try:
        results = vector_store.similarity_search(query, k=k)
        
        logger.info(f"\n===== Resultados para la búsqueda: '{query}' =====")
        for i, result in enumerate(results):
            logger.info(f"\nResultado {i+1}:")
            logger.info(f"Fuente: {result.metadata['source']}")
            logger.info(f"Título: {result.metadata['title']}")
            logger.info(f"Contenido: {result.page_content[:150]}...")
        
        return results
    except Exception as e:
        logger.error(f"Error al realizar búsqueda de prueba: {e}")
        return []


def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Procesa documentos y crea un vector store para búsquedas semánticas."
    )
    
    parser.add_argument(
        "--json-path", 
        type=str, 
        default=os.path.join("data", "all_documents.json"),
        help="Ruta al archivo JSON con los documentos"
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=DEFAULT_CHUNK_SIZE,
        help="Tamaño de los chunks en caracteres"
    )
    
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=DEFAULT_CHUNK_OVERLAP,
        help="Superposición entre chunks en caracteres"
    )
    
    parser.add_argument(
        "--embedding-model", 
        type=str, 
        default=DEFAULT_EMBEDDING_MODEL,
        help="Modelo de embeddings a utilizar"
    )
    
    parser.add_argument(
        "--store-type", 
        type=str, 
        default="faiss",
        choices=["faiss", "chroma"],
        help="Tipo de vector store a crear"
    )
    
    parser.add_argument(
        "--test-query", 
        type=str, 
        default="Cómo pago mi tarjeta Tuya",
        help="Consulta de prueba para verificar el vector store"
    )
    
    parser.add_argument(
        "--openai-api-key", 
        type=str, 
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key para OpenAI (requerida para modelos de OpenAI)"
    )
    
    return parser.parse_args()


def main():
    """Función principal del script."""
    args = parse_arguments()
    
    # Crear configuración
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        store_type=args.store_type
    )
    
    # Ruta al archivo JSON con los documentos
    json_path = args.json_path
    
    # Cargar documentos
    logger.info("Cargando documentos...")
    documents = load_documents_from_json(json_path)
    logger.info(f"Se cargaron {len(documents)} documentos")
    
    # Procesar documentos
    logger.info("Procesando documentos...")
    processed_docs = process_documents(documents)
    logger.info(f"Se procesaron {len(processed_docs)} documentos")
    
    # Dividir en chunks
    logger.info("Dividiendo documentos en chunks...")
    chunks = split_documents(processed_docs, config)
    logger.info(f"Se crearon {len(chunks)} chunks")
    
    # Crear vector store
    logger.info(f"Creando vector store con {config.store_type}...")
    vector_store = create_vector_store(chunks, config)
    logger.info(f"Vector store {config.store_type} creado exitosamente")
    
    # Ejemplo de búsqueda
    perform_test_query(vector_store, query=args.test_query)
    
    logger.info("\n¡Proceso completado! Los documentos están listos para ser consultados.")


if __name__ == "__main__":
    main()