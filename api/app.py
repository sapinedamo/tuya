"""
API de FastAPI para el sistema de consultas RAG de Tuya.

Este módulo implementa una API REST que permite realizar consultas al sistema
de recuperación y generación de respuestas de Tuya.
"""
import os
import sys
import torch
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Asegurarse que el módulo src pueda importarse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import TuyaRetriever
from src.generator import ResponseGenerator

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Modelos Pydantic para la API
class QueryRequest(BaseModel):
    """Modelo para la solicitud de consulta."""
    query: str
    max_documents: Optional[int] = 4
    model_id: Optional[str] = "unsloth/Llama-3.2-1B-Instruct"
    index_id: Optional[str] = None  # ID del índice a usar
    index_path: Optional[str] = None  # Ruta al índice
    openai_api_key: Optional[str] = None  # API key para OpenAI (tanto para embeddings como para LLM)

class DocumentInfo(BaseModel):
    """Información sobre un documento recuperado."""
    content: str
    source: Optional[str] = None
    title: Optional[str] = None

class QueryResponse(BaseModel):
    """Modelo para la respuesta a una consulta."""
    query: str
    answer: str
    documents: List[DocumentInfo]

# Nuevos modelos para indexación
class IndexingRequest(BaseModel):
    """Modelo para solicitar una indexación."""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    store_type: str = "faiss"
    openai_api_key: Optional[str] = None

class IndexingResponse(BaseModel):
    """Modelo para la respuesta de indexación."""
    status: str
    message: str
    index_id: str

# Variables globales para almacenar las instancias de retriever y generator
retriever = None
generator = None

# Variables para seguimiento de tareas de indexación
indexing_tasks = {}

# Lifespan moderno (reemplaza on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar
    global retriever, generator
    
    # Verificar disponibilidad de CUDA
    cuda_available = torch.cuda.is_available()
    
    try:
        # Inicializar retriever
        retriever = TuyaRetriever(
            vector_store_type="faiss",
            vector_store_path="faiss_index",
            embedding_model="all-MiniLM-L6-v2",
            top_k=4
        )
        
        # Inicializar generator (sin cuantización si no hay CUDA)
        generator = ResponseGenerator(
            model_id="unsloth/Llama-3.2-1B-Instruct",
            use_local=True,
            quantize=cuda_available,
            device="cuda" if cuda_available else "cpu",
            models_cache_dir="models"
        )
        print("✅ Sistema RAG inicializado correctamente")
    except Exception as e:
        print(f"❌ Error al inicializar el sistema RAG: {e}")
    
    yield

# Inicializar FastAPI con el lifespan
app = FastAPI(
    title="API de Consultas Tuya",
    description="API para consultar la base de conocimiento y generar respuestas.",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción para restringir orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para manejar errores de forma global
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error interno del servidor: {str(e)}"}
        )

@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Realiza una consulta a la base de conocimiento y genera una respuesta.
    """
    global retriever, generator
    
    try:
        # Si se especifica un índice personalizado, crear un nuevo retriever
        if request.index_id or request.index_path:
            # Determinar tipo y ruta del índice
            if request.index_id:
                # Buscar en índices conocidos
                if os.path.exists(f"faiss_indexes/{request.index_id}"):
                    index_type = "faiss"
                    index_path = f"faiss_indexes/{request.index_id}"
                elif os.path.exists(f"chroma_indexes/{request.index_id}"):
                    index_type = "chroma"
                    index_path = f"chroma_indexes/{request.index_id}"
                else:
                    raise HTTPException(status_code=404, detail=f"Índice {request.index_id} no encontrado")
            else:
                # Usar ruta proporcionada
                index_path = request.index_path
                index_type = "faiss" if "faiss" in index_path.lower() else "chroma"
            
            # Determinar modelo de embeddings
            if "text-embedding-" in request.index_id:
                # Para modelos de OpenAI mantener el formato original
                embedding_model = request.index_id.split("_")[0]
            else:
                embedding_model = "all-MiniLM-L6-v2"  # Default
                
            # Crear retriever específico
            custom_retriever = TuyaRetriever(
                vector_store_type=index_type,
                vector_store_path=index_path,
                embedding_model=embedding_model,
                top_k=request.max_documents,
                openai_api_key=request.openai_api_key
            )
            
            # Usar este retriever para la consulta actual
            current_retriever = custom_retriever
        else:
            # Usar retriever global
            if not retriever:
                raise HTTPException(
                    status_code=503, 
                    detail="El sistema de consulta no está inicializado correctamente."
                )
            
            # Actualizar número de documentos si es necesario
            if request.max_documents != retriever.top_k:
                retriever.top_k = request.max_documents
                retriever.retriever = retriever._create_retriever()
                
            current_retriever = retriever
            
        # Verificar si necesitamos un generador específico para esta consulta
        current_generator = generator
        if request.model_id and request.model_id != "unsloth/Llama-3.2-1B-Instruct":
            # Crear un generador personalizado si se especifica un modelo diferente
            is_openai_model = request.model_id.startswith("gpt-")
            
            # Verificar API key para modelos de OpenAI
            if is_openai_model and not request.openai_api_key:
                raise HTTPException(
                    status_code=400, 
                    detail="Se requiere OpenAI API key para modelos de OpenAI"
                )
            
            logger.info(f"Creando generador personalizado con modelo: {request.model_id}")
            current_generator = ResponseGenerator(
                model_id=request.model_id,
                use_local=not is_openai_model,  # Local solo si no es OpenAI
                quantize=not is_openai_model and torch.cuda.is_available(),
                device="cuda" if torch.cuda.is_available() else "cpu",
                models_cache_dir="models",
                openai_api_key=request.openai_api_key
            )
        
        # Recuperar documentos relevantes
        documents = current_retriever.retrieve(request.query)
        
        # Obtener contexto para generar la respuesta
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Generar respuesta con el generador correspondiente
        response = current_generator.generate_response(request.query, context)
        
        # Preparar información de documentos para la respuesta
        doc_info = []
        for doc in documents:
            doc_info.append(DocumentInfo(
                content=doc.page_content,
                source=doc.metadata.get("source", None),
                title=doc.metadata.get("title", None)
            ))
        
        return QueryResponse(
            query=request.query,
            answer=response,
            documents=doc_info
        )
        
    except Exception as e:
        logger.error(f"Error en consulta: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la consulta: {str(e)}"
        )

@app.post("/api/index", response_model=IndexingResponse)
async def index_documents(request: IndexingRequest, background_tasks: BackgroundTasks):
    """
    Crea un nuevo índice con el modelo de embedding especificado.
    
    - **embedding_model**: Modelo de embeddings ("all-MiniLM-L6-v2", "text-embedding-3-large", etc.)
    - **chunk_size**: Tamaño de los chunks en caracteres
    - **chunk_overlap**: Superposición entre chunks
    - **store_type**: Tipo de almacenamiento vectorial ("faiss" o "chroma")
    - **openai_api_key**: API key de OpenAI para modelos de OpenAI
    """
    # Validar si es un modelo de OpenAI y tiene API key
    if request.embedding_model.startswith("text-embedding-") and not request.openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="Se requiere OpenAI API key para modelos de embeddings de OpenAI"
        )
    
    # Generar ID único para este índice
    import hashlib
    import time
    
    model_id = request.embedding_model.replace("/", "-")
    timestamp = int(time.time())
    index_id = f"{model_id}_{timestamp}"
    
    # Iniciar tarea de indexación en segundo plano
    indexing_tasks[index_id] = {"status": "processing", "message": "Indexación en progreso"}
    
    # Lanzar tarea en segundo plano
    json_path = os.path.join("data", "all_documents.json")
    background_tasks.add_task(
        index_documents_task,
        request.embedding_model,
        request.chunk_size,
        request.chunk_overlap,
        request.store_type,
        json_path,
        index_id,
        request.openai_api_key
    )
    
    return IndexingResponse(
        status="processing",
        message="Indexación iniciada en segundo plano",
        index_id=index_id
    )

@app.get("/api/index/{index_id}/status")
async def get_indexing_status(index_id: str):
    """Obtiene el estado de una tarea de indexación."""
    if index_id not in indexing_tasks:
        raise HTTPException(status_code=404, detail="Tarea de indexación no encontrada")
    
    return indexing_tasks[index_id]

@app.get("/api/indexes")
async def list_indexes():
    """Lista todos los índices disponibles."""
    indexes = []
    
    # Buscar índices FAISS
    if os.path.exists("faiss_indexes"):
        faiss_dirs = os.listdir("faiss_indexes")
        for dir_name in faiss_dirs:
            if os.path.exists(os.path.join("faiss_indexes", dir_name, "index.faiss")):
                indexes.append({
                    "id": dir_name,
                    "type": "faiss",
                    "path": f"faiss_indexes/{dir_name}"
                })
    
    # Buscar índices Chroma
    if os.path.exists("chroma_indexes"):
        chroma_dirs = os.listdir("chroma_indexes")
        for dir_name in chroma_dirs:
            if os.path.isdir(os.path.join("chroma_indexes", dir_name)):
                indexes.append({
                    "id": dir_name,
                    "type": "chroma",
                    "path": f"chroma_indexes/{dir_name}"
                })
    
    return {"indexes": indexes}

@app.get("/api/health")
async def health_check():
    """Endpoint de verificación de salud de la API."""
    return {
        "status": "ok", 
        "cuda_available": torch.cuda.is_available(),
        "retriever_ready": retriever is not None,
        "generator_ready": generator is not None
    }

@app.get("/")
async def root():
    """Endpoint raíz que redirige a la documentación."""
    return {"message": "API de consultas Tuya activa. Visita /docs para la documentación."}

# Si quieres servir una interfaz web estática
if os.path.exists("static"):
    app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

# Función de indexación en segundo plano
def index_documents_task(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int, 
    store_type: str,
    json_path: str,
    index_id: str,
    openai_api_key: Optional[str] = None
):
    """Ejecuta la indexación en segundo plano."""
    try:
        # Guardar API key temporal si se proporciona
        original_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
        from vectorize_data import load_documents_from_json, process_documents, split_documents, create_vector_store
        from dataclasses import dataclass
        
        @dataclass
        class IndexConfig:
            chunk_size: int
            chunk_overlap: int
            embedding_model: str
            store_type: str
            openai_api_key: Optional[str] = None  # Añadir para pasar explícitamente
            
        # Crear directorio para el índice si no existe
        os.makedirs(f"{store_type}_indexes/{index_id}", exist_ok=True)
            
        # Cargar y procesar documentos
        documents = load_documents_from_json(json_path)
        processed_docs = process_documents(documents)
        
        # Configuración para la indexación
        config = IndexConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            store_type=store_type,
            openai_api_key=openai_api_key  # Pasar la API key a la configuración
        )
        
        # Dividir en chunks
        chunks = split_documents(processed_docs, config)
        
        # Crear vector store
        index_path = f"{store_type}_indexes/{index_id}"
        vector_store = create_vector_store(chunks, config, output_dir=index_path)
        
        # Actualizar estado de la tarea
        indexing_tasks[index_id] = {"status": "completed", "message": "Indexación completada con éxito"}
        
        # Restaurar API key original
        if openai_api_key and original_api_key:
            os.environ["OPENAI_API_KEY"] = original_api_key
        elif openai_api_key:
            del os.environ["OPENAI_API_KEY"]
            
    except Exception as e:
        indexing_tasks[index_id] = {"status": "failed", "message": f"Error en indexación: {str(e)}"}
        logger.error(f"Error en indexación {index_id}: {e}")

if __name__ == "__main__":
    import uvicorn
    # El comando correcto para ejecutar este archivo
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)