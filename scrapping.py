import requests
from bs4 import BeautifulSoup
import time
import random
import json
import os
from datetime import datetime
import hashlib

def scrap_pages():
    urls = [
        "https://www.tuya.com.co/como-pago-mi-tarjeta-o-credicompras",
        "https://www.tuya.com.co/tarjetas-de-credito",
        "https://www.tuya.com.co/credicompras",
        "https://www.tuya.com.co/otras-soluciones-financieras",
        "https://www.tuya.com.co/nuestra-compania",
        "https://www.tuya.com.co/activacion-tarjeta"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Sec-Ch-Ua": "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\"",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }
    
    content_list = []
    session = requests.Session()
    
    # Crear directorio para guardar los archivos si no existe
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Primero visitamos la página principal para obtener cookies
    try:
        session.get("https://www.tuya.com.co/", headers=headers, timeout=10)
    except Exception as e:
        print(f"Error al visitar la página principal: {e}")
    
    for url in urls:
        try:
            # Añadimos un retraso aleatorio entre solicitudes para simular comportamiento humano
            time.sleep(random.uniform(1, 3))
            
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraer título
            title = soup.title.string if soup.title else "Sin título"
            
            # Extraer contenido principal
            page_text = soup.get_text(separator="\n", strip=True)
            
            # Crear ID único para el documento
            doc_id = hashlib.md5(url.encode()).hexdigest()
            
            # Crear diccionario con el contenido y metadatos
            document = {
                "id": doc_id,
                "url": url,
                "title": title,
                "content": page_text,
                "source": "web_scraping",
                "timestamp": datetime.now().isoformat()
            }
            
            content_list.append(document)
            
            # Guardar individualmente cada documento
            filename = f"{doc_id}.json"
            with open(os.path.join(data_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            
            print(f"Éxito al extraer datos de: {url}")
        except Exception as e:
            error_doc = {
                "id": hashlib.md5(url.encode()).hexdigest(),
                "url": url,
                "content": f"Error al extraer datos: {e}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            content_list.append(error_doc)
            print(f"Error en URL {url}: {e}")
    
    # Guardar todos los documentos en un solo archivo
    with open(os.path.join(data_dir, "all_documents.json"), 'w', encoding='utf-8') as f:
        json.dump(content_list, f, ensure_ascii=False, indent=2)
    
    return content_list

if __name__ == "__main__":
    scraped_data = scrap_pages()
    print(f"\nSe han guardado {len(scraped_data)} documentos en la carpeta 'data'")
    print("Los documentos están listos para ser cargados en un vector store")