import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Mystique03/iris-qwen3-4b-gguf",
    filename="qwen3-4b-instruct-2507.Q4_K_M.gguf",
)

llm = Llama(
    model_path=model_path,
    n_ctx=1024,
    n_threads=4,
    verbose=True,
)

def respond(prompt: str) -> str:
    try:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Answer the patient's concern clearly and carefully."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7,
        )
        return output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=10, label="Enter your concern here"),
    outputs=gr.Textbox(lines=10, label="Response"),
    title="Iris - Medical Assistant",
    description="Ask Iris for medical advice.",
    share=True,
).launch(server_name="0.0.0.0", server_port=7860)
