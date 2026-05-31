""" Hugging Face Space for Iris!"""

import threading
import time
from collections import deque

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
 
from iris.security import validate_output, validate_user_input

# Model Loading

model_path = hf_hub_download(
    repo_id="Mystique03/iris-qwen3-4b-gguf",
    filename="iris-qwen3-4b-merged.Q4_K_M.gguf",
)

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
)

SYSTEM_PROMPT = (
    "You are Iris, a medical assistant providing preliminary symptom assessments. "
    "Always give a specific assessment based on the symptoms given. "
    "Provide: "
    "1. Most likely condition(s) with reasoning, "
    "2. Recommended next steps and home care, "
    "3. Red flag symptoms that need immediate attention. "
    "Be specific and helpful. Never refuse to assess."
)

MAX_REQUESTS_PER_MINUTE = 10
request_times: deque = deque()
request_lock = threading.Lock()

def check_rate_limits():
    now = time.time()
    with request_lock:
        while request_times and request_times[0] < now - 60:
            request_times.popleft()
        if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
            return False
        request_times.append(now)
        return True
    
# Gradio handeler

def respond(prompt: str) -> str:
    """Validate input, run inference, and return a safe response."""
    if not check_rate_limits():
        return "Too many requests. Please wait a moment and try again."
 
    cleaned, err = validate_user_input(prompt)
    if err:
        return f"Input error: {err}"
 
    try:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": cleaned},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        result = output["choices"][0]["message"]["content"].strip()
        return validate_output(result)
 
    except Exception as e:
        return f"Error: {e}"
    
gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=6, label="Describe your symptoms"),
    outputs=gr.Textbox(lines=10, label="Iris"),
    title="Iris — Medical Assistant",
    description="Describe your symptoms and Iris will provide a preliminary assessment.",
).launch(server_name="0.0.0.0", server_port=7860)
    
