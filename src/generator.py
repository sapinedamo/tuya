"""
Módulo para generar respuestas basadas en el contexto recuperado.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Literal, Union
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from langchain_core.language_models import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
import torch

# Añadir importación para OpenAI
from langchain_openai import ChatOpenAI

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generator")

# Prompts por modelo
PROMPT_TEMPLATES = {
    "llama": """
<s>[INST]
Por favor responde la siguiente pregunta usando solo la información proporcionada en el contexto.
Si no tienes suficiente información en el contexto, responde "No tengo suficiente información para responder esta pregunta."

PREGUNTA: {question}

CONTEXTO:
{context}
[/INST]
""",
    
    "tinyllama": """
<|im_start|>system
Eres un asistente útil que responde preguntas basado únicamente en el contexto proporcionado.
<|im_end|>
<|im_start|>user
CONTEXTO:
{context}

PREGUNTA: {question}
<|im_end|>
<|im_start|>assistant
""",

    "openai": """
Eres un asistente especializado para Tuya S.A. Tu tarea es responder preguntas basándote únicamente en la información proporcionada en el contexto.
Si la información en el contexto no es suficiente para responder, debes decir "No tengo suficiente información para responder esta pregunta."

CONTEXTO:
{context}

PREGUNTA: {question}
"""
}

# Modelos disponibles predefinidos
AVAILABLE_MODELS = {
    "llama3": {
        "model_id": "unsloth/Llama-3.2-1B-Instruct", 
        "prompt_template": "llama",
        "output_marker": "[/INST]",
        "type": "local"
    },
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        "prompt_template": "tinyllama",
        "output_marker": "<|im_end|>",
        "type": "local"
    },
    "gpt-4o": {
        "model_id": "gpt-4o", 
        "prompt_template": "openai",
        "output_marker": None,
        "type": "api"
    }
}

class ResponseGenerator:
    """Generador de respuestas basado en modelos de lenguaje."""
    
    def __init__(
        self,
        model_id: str = "unsloth/Llama-3.2-1B-Instruct",
        model_variant: Optional[str] = None,
        use_local: bool = True,
        quantize: bool = True,
        device: str = "cpu",
        models_cache_dir: str = "models",
        hf_token: Optional[str] = None,
        openai_api_key: Optional[str] = None  # Nueva opción para OpenAI
    ):
        """
        Inicializa el generador de respuestas.
        
        Args:
            model_id: Identificador del modelo en HuggingFace/OpenAI o nombre corto.
            model_variant: Variante del modelo (llama3, tinyllama, gpt-4o, etc).
            use_local: Si es True, carga el modelo localmente (solo para modelos HF).
            quantize: Si es True, usa cuantización de 4 bits con BitsAndBytes.
            device: Dispositivo a usar ('cpu' o 'cuda').
            models_cache_dir: Directorio donde se descargarán los modelos.
            hf_token: Token opcional de HuggingFace para modelos que lo requieren.
            openai_api_key: API key para modelos de OpenAI.
        """
        # Verificar si se está usando un modelo predefinido
        if model_id in AVAILABLE_MODELS:
            self.model_variant = model_id
            self.model_id = AVAILABLE_MODELS[model_id]["model_id"]
            self.model_type = AVAILABLE_MODELS[model_id]["type"]
        elif model_variant in AVAILABLE_MODELS:
            self.model_variant = model_variant
            self.model_id = model_id
            self.model_type = AVAILABLE_MODELS[model_variant]["type"]
        else:
            # Si es un modelo de OpenAI
            if model_id.startswith("gpt-"):
                self.model_variant = "gpt-4o"  # Default para OpenAI
                self.model_id = model_id
                self.model_type = "api"
            else:
                # Si no es un modelo predefinido, usar configuración de llama3 por defecto
                self.model_variant = "llama3"
                self.model_id = model_id
                self.model_type = "local"
                
        self.use_local = use_local
        self.quantize = quantize
        self.device = device
        self.models_cache_dir = Path(models_cache_dir)
        self.hf_token = hf_token
        self.openai_api_key = openai_api_key
        
        # Configurar prompt según el modelo
        prompt_template = PROMPT_TEMPLATES.get(
            AVAILABLE_MODELS.get(self.model_variant, {}).get("prompt_template", "llama")
        )
        self.output_marker = AVAILABLE_MODELS.get(self.model_variant, {}).get("output_marker", "[/INST]")
        
        # Solo crear directorio de caché si se usa modelos locales
        if self.model_type == "local":
            # Crear directorio de cache si no existe
            if not self.models_cache_dir.exists():
                self.models_cache_dir.mkdir(parents=True)
            
            # Descargar modelo si es necesario
            if use_local and not os.path.exists(os.path.join(self.models_cache_dir, self.model_id.split('/')[-1])):
                self._download_model()
        
        self.llm = self._load_llm()
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.qa_chain = self._create_qa_chain()
        
        logger.info(f"Modelo inicializado: {self.model_id} (variante: {self.model_variant}, tipo: {self.model_type})")
    
    def _download_model(self):
        """Descarga el modelo desde HuggingFace."""
        try:
            logger.info(f"Descargando modelo {self.model_id}...")
            model_name = self.model_id.split('/')[-1]
            model_path = os.path.join(self.models_cache_dir, model_name)
            
            # Descargar directorio del modelo
            snapshot_download(
                repo_id=self.model_id,
                local_dir=model_path,
                token=self.hf_token,
                local_dir_use_symlinks=False
            )
            logger.info(f"Modelo descargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al descargar modelo: {e}")
            raise
    
    def _load_llm(self) -> LLM:
        """Carga el modelo de lenguaje."""
        try:
            # Si es un modelo de OpenAI
            if self.model_type == "api":
                if self.model_id.startswith("gpt-"):
                    if not self.openai_api_key:
                        # Intentar obtenerla del ambiente
                        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
                        if not self.openai_api_key:
                            raise ValueError("Se requiere OpenAI API key para modelos GPT")
                    
                    logger.info(f"Usando modelo de OpenAI: {self.model_id}")
                    llm = ChatOpenAI(
                        model=self.model_id,
                        openai_api_key=self.openai_api_key,
                        temperature=0.2,
                        max_tokens=1000
                    )
                    return llm
                else:
                    raise ValueError(f"Tipo de modelo API no soportado: {self.model_id}")
            
            # Para modelos locales (HuggingFace)
            model_path = self.model_id
            # Si es local, usar la ruta en disco
            if self.use_local:
                model_path = os.path.join(self.models_cache_dir, self.model_id.split('/')[-1])
            
            logger.info(f"Cargando modelo desde: {model_path}")
            
            # Configuración de cuantización
            quantization_config = None
            if self.quantize and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Cargar tokenizer y modelo
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Opción para CPU con poca memoria
            if self.device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    offload_folder="offload"
                )
            else:
                # Código para GPU sin cambios
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.quantize else torch.float32,
                    quantization_config=quantization_config,
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )
            
            # Crear pipeline con configuración para el modelo específico
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True
            )
            
            # Crear LLM para LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise

    def _create_qa_chain(self):
        """Crea la cadena de procesamiento para preguntas y respuestas."""
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate_response(self, question: str, context: str) -> str:
        """
        Genera una respuesta basada en la pregunta y el contexto.
        
        Args:
            question: Pregunta del usuario.
            context: Contexto relevante para responder.
            
        Returns:
            Respuesta generada.
        """
        try:
            if not context:
                return "No encontré información relevante para responder a tu pregunta."
                
            logger.info(f"Generando respuesta para: '{question}'")
            response = self.qa_chain.invoke({"context": context, "question": question})
            
            # Para modelos API como OpenAI no necesitamos limpieza
            if self.model_type == "api":
                return response.strip()
                
            # Limpiar posible texto adicional según el formato del modelo
            cleaned_response = response
            if self.output_marker and self.output_marker in response:
                cleaned_response = response.split(self.output_marker)[1]
                
            return cleaned_response.strip()
        except Exception as e:
            logger.error(f"Error al generar respuesta: {e}")
            return "Lo siento, ocurrió un error al generar la respuesta."