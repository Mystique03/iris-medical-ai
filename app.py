import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id = "Mystique03/iris-qwen3-4b-gguf",
    filename = "qwen3-4b-instruct-2507.Q4_K_M.gguf",
)

llm = Llama(
    model_path = model_path,
    n_ctx = 1024,
    n_threads = 4,
    verbose = False,
)

def respond(prompt: str) -> str:
    output = llm.create_chat_completion(
        messages = [
            {"role": "user", "content": prompt}
        ],
        max_tokens = 250,
        temperature = 0.7,
    )
    return output.choices[0].message.content.strip()

gr.Interface(
    fn = respond,
    inputs = gr.Textbox(lines=10, label="Enter your concern here"),
    outputs = gr.Textbox(lines=10, label="Response"),
    title = "Iris - Medical Assistant",
    description = "Ask Iris for medical advice based on symptoms and predictions.",
).launch()
