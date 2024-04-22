# https://github.com/ggerganov/llama.cpp project primarily supports model loading and inference on the CPU.
# https://github.com/abetlen/llama-cpp-python/
# pip install llama-cpp-python
# huggingface-cli download Qwen/CodeQwen1.5-7B-Chat-GGUF codeqwen-1_5-7b-chat-q8_0.gguf --local-dir . --local-dir-use-symlinks False
# move it to "C:\Users\dpolzer\.cache\huggingface\hub\models--Qwen--CodeQwen1.5-7B-Chat-GGUF\snapshots\" folder

# Tokenizer download (For gguf format you dont need these since its's a single file distribution)
# huggingface-cli download Qwen/CodeQwen1.5-7B-Chat tokenizer.json
# huggingface-cli download Qwen/CodeQwen1.5-7B-Chat tokenizer_config.json
# huggingface-cli download Qwen/CodeQwen1.5-7B-Chat vocab.txt

#Artifacts will end up in the 
#   [C:\Users\dpolzer\.cache\huggingface\hub\models--Qwen--CodeQwen1.5-7B-Chat\snapshots]

import llama_cpp
import llama_cpp.llama_tokenizer

# Initialize the model and tokenizer
llama = llama_cpp.Llama.from_pretrained(
    repo_id="Qwen/CodeQwen1.5-7B-Chat-GGUF",
    filename="codeqwen-1_5-7b-chat-q8_0.gguf",  # Updated to the quantized model file
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B"),
    verbose=False
)

# Define the prompt
prompt = "Write a quicksort algorithm in C#."

prompt2 = """
How would i rewrite this python 'distilbert/distilbert-base-uncased' model inference in C# and onnx ?

```python
import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="flash_attention_2")

text = "Replace me by any text you'd like."

encoded_input = tokenizer(text, return_tensors='pt').to(device)
model.to(device)

output = model(**encoded_input)
```"""

# Define the messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

print("Loaded the model, starting inference ... \n")

# Create the chat completion
response = llama.create_chat_completion(
    messages=messages,
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"}
            },
            "required": ["content"],
        }
    },
    stream=True
)

# Print the response
for chunk in response:
    delta = chunk["choices"][0]["delta"]
    if "content" not in delta:
        continue
    print(delta["content"], end="", flush=True)

print()
