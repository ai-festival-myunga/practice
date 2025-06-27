#%%
import  os
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(provider="hf-inference", model="meta-llama/Llama-3.3-70B-Instruct")

output = client.text_generation(
    "The capital of France is",
    max_new_tokens=100,
)

print(output)
# %%
import  os
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(provider="hf-inference", model="meta-llama/Llama-3.3-70B-Instruct")

prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
output = client.text_generation(
    prompt,
    max_new_tokens=100,
)

print(output)
# %%
import  os
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(provider="hf-inference", model="meta-llama/Llama-3.3-70B-Instruct")

output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)
# %%
