import os
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(model="Mystique03/medical-qwen3-4b-lora", token=os.environ.get("HF_TOKEN"))

def respond(prompt: str) -> str:
    output = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.7,
    )
    return output.choices[0].message.content.strip()

gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=10, label="Enter your concern here"),
    outputs=gr.Textbox(lines=10, label="Response"),
    title="Iris - Medical Assistant",
    description="Ask Iris for medical advice based on symptoms and predictions.",
).launch()
