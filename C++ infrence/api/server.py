from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import tiktoken
import os
import sys

app = FastAPI()

# Allow Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str

# Load Tokenizer
try:
    enc = tiktoken.get_encoding("gpt2")
except:
    print("Error: tiktoken not installed. Run 'pip install tiktoken'")

@app.post("/generate")
async def generate_sql(request: QueryRequest):
    full_prompt = f"USER: {request.text}\nASSISTANT:"
    print(f"DEBUG: Processing: {full_prompt}")
    
    tokens = enc.encode(full_prompt)
    token_str = ",".join(map(str, tokens))
    
    # --- PATH SETUP (Connecting API to Inference) ---
    
    # 1. Find the 'inference' directory (Go up one level from 'api')
    current_dir = os.path.dirname(os.path.abspath(__file__)) # inside /api
    project_root = os.path.dirname(current_dir)              # inside /project
    inference_dir = os.path.join(project_root, "inference")  # inside /project/inference
    
    # 2. Define Executable Path
    exe_name = "inference.exe" if os.name == 'nt' else "inference"
    exe_path = os.path.join(inference_dir, exe_name)
    
    # Check if exists
    if not os.path.exists(exe_path):
        return {"result": f"Error: {exe_name} not found in {inference_dir}. Did you compile it?"}

    # 3. Run Engine (Crucial: Set cwd=inference_dir so it finds model.bin)
    try:
        process = subprocess.run(
            [exe_path, token_str], 
            capture_output=True, 
            text=True,
            cwd=inference_dir  # <--- Engine will run INSIDE inference folder
        )
        
        output = process.stdout
        if not output:
            output = f"Error (Stderr): {process.stderr}"
            
    except Exception as e:
        output = f"Server Error: {str(e)}"

    return {"result": output}