"""
Script para consultar la base de conocimiento y generar respuestas.
"""
import argparse
import logging
import sys
import os
import torch

from src.retriever import TuyaRetriever
from src.generator import ResponseGenerator

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query_engine")

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Consulta la base de conocimiento y genera respuestas."
    )
    
    parser.add_argument(
        "--query", 
        type=str,
        help="Consulta a realizar"
    )
    
    parser.add_argument(
        "--vector-store-type", 
        type=str, 
        default="faiss",
        choices=["faiss", "chroma"],
        help="Tipo de vector store a utilizar"
    )
    
    parser.add_argument(
        "--vector-store-path", 
        type=str, 
        default="faiss_index",
        help="Ruta al vector store"
    )
    
    parser.add_argument(
        "--embedding-model", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="Modelo de embeddings a utilizar"
    )
    
    parser.add_argument(
        "--llm-model", 
        type=str, 
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Modelo de lenguaje a utilizar"
    )
    
    parser.add_argument(
        "--use-local", 
        action="store_true",
        help="Usar modelo local en lugar de API"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Modo interactivo para múltiples consultas"
    )
    
    parser.add_argument(
        "--quantize", 
        action="store_true",
        default=True,
        help="Usar cuantización de 4 bits con BitsAndBytes"
    )

    parser.add_argument(
        "--models-cache-dir", 
        type=str,
        default="models",
        help="Directorio donde se guardarán los modelos descargados"
    )

    parser.add_argument(
        "--hf-token", 
        type=str,
        default=os.environ.get("HUGGINGFACE_TOKEN", None),
        help="Token de HuggingFace para modelos que lo requieren"
    )

    parser.add_argument(
        "--no-quantize", 
        action="store_true",
        help="Desactivar cuantización (recomendado para CPU)"
    )
    
    parser.add_argument(
        "--model-variant", 
        type=str, 
        default=None,
        choices=["llama3", "tinyllama"],
        help="Variante específica de modelo a utilizar"
    )
    
    parser.add_argument(
        "--index-path",
        type=str,
        default="faiss_index",
        help="Ruta al índice vectorial a utilizar"
    )

    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key para OpenAI Embeddings"
    )
    
    return parser.parse_args()

def interactive_mode(retriever: TuyaRetriever, generator: ResponseGenerator):
    """Ejecuta el modo interactivo para consultas múltiples."""
    print("\n===== Consulta de Información Tuya =====")
    print("Escribe 'salir' o 'exit' para terminar.\n")
    
    while True:
        try:
            query = input("\nPregunta: ").strip()
            if query.lower() in ["salir", "exit", "quit"]:
                print("\n¡Hasta pronto!")
                break
                
            if not query:
                continue
                
            print("\nBuscando información relevante...")
            context = retriever.get_relevant_context(query)
            
            print("Generando respuesta...\n")
            response = generator.generate_response(query, context)
            
            print(f"Respuesta: {response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta pronto!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Ocurrió un error: {e}")

def main():
    """Función principal del script."""
    args = parse_arguments()

    if args.no_quantize:
        args.quantize = False
    
    try:
        # Verificar si CUDA está disponible
        cuda_available = torch.cuda.is_available()
        
        # Si CUDA no está disponible, desactivar cuantización
        if not cuda_available and args.quantize:
            logger.warning("CUDA no disponible. Desactivando cuantización automáticamente.")
            args.quantize = False
        
        # Inicializar el retriever
        retriever = TuyaRetriever(
            vector_store_type=args.vector_store_type,
            vector_store_path=args.vector_store_path,
            embedding_model=args.embedding_model
        )
        
        # Inicializar el generador de respuestas
        generator = ResponseGenerator(
            model_id=args.llm_model,
            model_variant=args.model_variant,
            use_local=args.use_local,
            quantize=args.quantize,
            device="cuda" if cuda_available else "cpu",
            models_cache_dir=args.models_cache_dir,
            hf_token=args.hf_token
        )
        
        if args.interactive:
            interactive_mode(retriever, generator)
        elif args.query:
            # Recuperar contexto relevante
            context = retriever.get_relevant_context(args.query)
            
            # Generar respuesta
            response = generator.generate_response(args.query, context)
            
            print(f"\nPregunta: {args.query}\n")
            print(f"Respuesta: {response}\n")
        else:
            logger.error("Debes proporcionar una consulta o utilizar el modo interactivo")
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()