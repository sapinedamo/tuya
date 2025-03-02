"""
Módulo para recuperar información relevante del vector store.
"""
import os
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retriever")

class TuyaRetriever:
    """Retriever especializado para consultar la base de conocimiento de Tuya."""

    def __init__(
        self,
        vector_store_type: str = "faiss",
        vector_store_path: str = "faiss_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 4,
        openai_api_key: Optional[str] = None
    ):
        """
        Inicializa el retriever con el vector store especificado.
        
        Args:
            vector_store_type: Tipo de vector store ('faiss' o 'chroma').
            vector_store_path: Ruta al vector store.
            embedding_model: Modelo de embeddings a utilizar.
            top_k: Cantidad de documentos a recuperar.
            openai_api_key: API key para OpenAI (opcional).
        """
        self.vector_store_type = vector_store_type
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        
        # Configurar embeddings
        if embedding_model.startswith("text-embedding-"):
            # Usar OpenAI embeddings
            from langchain_openai import OpenAIEmbeddings
            
            if openai_api_key:
                self.embeddings = OpenAIEmbeddings(
                    model=embedding_model,
                    openai_api_key=openai_api_key
                )
            else:
                # Usar la variable de entorno OPENAI_API_KEY
                self.embeddings = OpenAIEmbeddings(model=embedding_model)
                
            logger.info(f"Usando embeddings de OpenAI: {embedding_model}")
        else:
            # Usar HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=f"sentence-transformers/{embedding_model}"
            )
            logger.info(f"Usando embeddings de HuggingFace: {embedding_model}")
            
        self.vector_store = self._load_vector_store()
        self.retriever = self._create_retriever()
        
    def _load_vector_store(self):
        """Carga el vector store desde el disco."""
        try:
            if self.vector_store_type.lower() == "faiss":
                if not os.path.exists(self.vector_store_path):
                    raise FileNotFoundError(f"No se encontró el vector store en {self.vector_store_path}")
                vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Vector store FAISS cargado desde {self.vector_store_path}")
                return vector_store
            
            elif self.vector_store_type.lower() == "chroma":
                if not os.path.exists(self.vector_store_path):
                    raise FileNotFoundError(f"No se encontró el vector store en {self.vector_store_path}")
                vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings
                )
                logger.info(f"Vector store Chroma cargado desde {self.vector_store_path}")
                return vector_store
            
            else:
                raise ValueError(f"Tipo de vector store no soportado: {self.vector_store_type}")
                
        except Exception as e:
            logger.error(f"Error al cargar el vector store: {e}")
            raise
            
    def _create_retriever(self) -> VectorStoreRetriever:
        """Crea un retriever a partir del vector store."""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        return retriever
        
    def retrieve(self, query: str) -> List[Document]:
        """
        Recupera documentos relevantes para la consulta.
        
        Args:
            query: Consulta del usuario.
            
        Returns:
            Lista de documentos relevantes.
        """
        try:
            documents = self.retriever.invoke(query)
            logger.info(f"Recuperados {len(documents)} documentos para la consulta: '{query}'")
            return documents
        except Exception as e:
            logger.error(f"Error al recuperar documentos: {e}")
            return []
            
    def get_relevant_context(self, query: str) -> str:
        """
        Obtiene el contexto relevante como texto concatenado.
        
        Args:
            query: Consulta del usuario.
            
        Returns:
            Contexto relevante concatenado.
        """
        documents = self.retrieve(query)
        if not documents:
            return ""
            
        context = "\n\n".join([doc.page_content for doc in documents])
        return context