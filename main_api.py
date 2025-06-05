from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from bs4 import BeautifulSoup
import time
# from requests.exceptions import RequestException
# from llama_index.llms.azure_openai import AzureOpenAI
# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from urllib.parse import urljoin, urlparse
import os
import json
import random
from dotenv import load_dotenv
import logging
import groq
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from supabase import create_client, Client


load_dotenv()


# load_dotenv()

app = FastAPI()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

url = os.environ['SUPABASE_URL']  # Your Supabase project URL
key = os.environ['SUPABASE_SERVICE_ROLE_KEY']  # Use service_role key for bypassing RLS
supabase: Client = create_client(url, key)


user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/102.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/103.0.1264.49",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 15_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; SM-A217F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Mobile Safari/537.36"
]


class RAGRequest(BaseModel):
    file_path: str
    prompt: str

class URL(BaseModel):
    url: str




@app.post("/rag")
async def rag(request: RAGRequest):
    try:
        # Import necessary packages
        
        print(os.environ['hf_token'])
        # HuggingFace Inference API for embeddings - using your working code
        API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction"
        headers = {
            "Authorization": os.environ['hf_token'],  
        }
        
        # Your working query function
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        
        # Create a Groq client
        groq_client = groq.Client(
            api_key=os.environ["groq_token"],
        )
        
        # Function to process text with Groq
        def process_with_groq(query_text, context):
            prompt = f"""
            Context information:
            {context}
            
            Based on the context information above, please answer the following question:
            {query_text}
            
            Answer:
            """
            
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.4,
                max_tokens=512
            )
            
            return response.choices[0].message.content
        
        # Function to get file content from Supabase
        def get_file_from_supabase(bucket_name, file_path):
            try:
                # Download file from Supabase storage
                response = supabase.storage.from_(bucket_name).download(file_path)
                
                # Decode bytes to string (assuming UTF-8 encoding for text files)
                content = response.decode('utf-8')
                return content
                
            except Exception as e:
                logger.error(f"Error downloading file from Supabase: {e}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"File not found in Supabase bucket: {file_path}"
                )
        
        # Get file content from Supabase instead of local file
        # Assuming request.file_path contains the bucket name and file path
        # Format: "bucket_name/file_path" or just "file_path" if bucket is fixed
        
        bucket_name = "url-2-ans-bucket"  # Your bucket name
        file_path = request.file_path  # This should be the file name/path in the bucket
        
        # Download and read the document from Supabase
        content = get_file_from_supabase(bucket_name, file_path)
        
        logger.info(f"Successfully downloaded file from Supabase: {file_path}")
        
        # Simple text chunking (adjust chunk size as needed)
        chunk_size = 1000
        overlap = 200
        chunks = []
        
        # Create overlapping chunks of text
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            if len(chunk) > 100:  # Only keep substantial chunks
                chunks.append({"text": chunk, "position": i})
        
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Get embeddings for all chunks
        chunk_embeddings = []
        for chunk in chunks:
            # Use your working query function
            embedding = query({"inputs": chunk["text"]})
            chunk_embeddings.append(embedding)
        
        # Get embedding for the query
        query_embedding = query({"inputs": request.prompt})
        
        # Calculate similarity between query and all chunks
        similarities = []
        for chunk_embedding in chunk_embeddings:
            # Convert to numpy arrays for similarity calculation
            query_np = np.array(query_embedding)
            chunk_np = np.array(chunk_embedding)
            
            # Reshape if needed
            if len(query_np.shape) == 1:
                query_np = query_np.reshape(1, -1)
            if len(chunk_np.shape) == 1:
                chunk_np = chunk_np.reshape(1, -1)
                
            # Calculate cosine similarity
            similarity = cosine_similarity(query_np, chunk_np)[0][0]
            similarities.append(similarity)
        
        # Get indices of top 3 most similar chunks
        top_k = 3
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Extract the most relevant chunks
        relevant_chunks = [chunks[i]["text"] for i in top_indices]
        context_text = "\n\n".join(relevant_chunks)
        
        # Process with Groq
        answer = process_with_groq(request.prompt, context_text)
        
        # Prepare sources
        sources = [{"text": chunks[i]["text"][:200] + "...", "position": chunks[i]["position"]} 
                  for i in top_indices]
        
        return {
            "sources": sources,
            "user_query": request.prompt,
            "assistant_response": answer,
            "file_source": f"supabase://{bucket_name}/{file_path}"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception("Error occurred in RAG process")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.post("/extract_links")
async def extract_links(url: URL):
    def extract_unique_links(url_string, max_retries=3, timeout=30):
        for attempt in range(max_retries):
            try:
               
                headers = {'User-Agent': random.choice(user_agents)}
                response = requests.get(url_string, headers=headers, timeout=timeout)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                base_url = urlparse(url_string)
                base_url = f"{base_url.scheme}://{base_url.netloc}"
                
                a_tags = soup.find_all('a', href=True)
                links = []
                for a in a_tags:
                    href = a.get('href')
                    
                    full_url = urljoin(base_url, href)
                    links.append(full_url)
                
                unique_links = list(dict.fromkeys(links))
                unique_links.insert(0, url_string)
                return unique_links
            
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to retrieve {url_string} after {max_retries} attempts.")
                    raise HTTPException(status_code=500, detail=f"Failed to retrieve {url_string} after {max_retries} attempts.")
        return []
    
    unique_links = extract_unique_links(url.url)
    return {"unique_links": unique_links}

@app.post("/extract_text")
async def extract_text(urls: List[str]):
    output_file = "extracted_text.txt"
    
    def upload_text_content(filename, content, bucket_name):
        try:
            # Convert string content to bytes
            file_content = content.encode('utf-8')
            
            # Upload to Supabase bucket with upsert parameter separate
            response = supabase.storage.from_(bucket_name).upload(
                path=filename,
                file=file_content,
                file_options={
                    "content-type": "text/plain"
                }
            )
            
            print(f"Text file uploaded successfully: {response}")
            return response
            
        except Exception as e:
            print(f"Error uploading text content: {e}")
            # If file already exists, try to update it
            try:
                response = supabase.storage.from_(bucket_name).update(
                    path=filename,
                    file=file_content,
                    file_options={
                        "content-type": "text/plain"
                    }
                )
                print(f"Text file updated successfully: {response}")
                return response
            except Exception as update_error:
                print(f"Error updating text content: {update_error}")
                return None
    
    def text_data_extractor(links):
        extracted_texts = []
        
        for link in links:
            parsed_url = urlparse(link)
            if parsed_url.scheme:
                full_url = link
            else:
                print("url invalid!")
                continue

            retries = 3
            while retries > 0:
                try:
                    
                    headers = {'User-Agent': random.choice(user_agents)}
                    response = requests.get(full_url, headers=headers, timeout=30)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text()
                    clean_text = ' '.join(text.split())
                    extracted_texts.append({"url": full_url, "text": clean_text})
                    
                    
                    # f.write(f"URL: {full_url}\n")
                    # f.write(f"Text: {clean_text}\n\n")
                    # f.flush() 
                    
                    break
                except requests.RequestException as e:
                    retries -= 1
                    logger.warning(f"Retry {3 - retries} for {full_url} failed: {e}")
                    if retries > 0:
                        wait_time = 5 * (3 - retries)
                        time.sleep(wait_time)
                
            if retries == 0:
                extracted_texts.append({"url": full_url, "text": "Failed to retrieve text after multiple attempts."})
                
                # f.write(f"URL: {full_url}\n")
                # f.write("Text: Failed to retrieve text after multiple attempts.\n\n")
                # f.flush()
        
        return extracted_texts
    
    extracted_data = text_data_extractor(urls)
    
    string_output = json.dumps(extracted_data)
    upload_text_content(output_file, string_output, "url-2-ans-bucket")
    
    return {"extracted_data": extracted_data, "file_saved": output_file}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
