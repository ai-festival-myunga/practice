#%%
import  os
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()

client = InferenceClient("meta-llama/Llama-3.3-70B-Instruct")

output = client.text_generation(
    "The capital of France is",
    max_new_tokens=100,
)

print(output)
# %%
