### Build / Run

```bash
# build 
cargo b

# examples --running inference on different models

#run jina-bert in release (gpu not supported > no cuda implementation for softmax-last-dim)
cargo run --release --bin jina-bert -- --cpu --prompt "The best thing about coding in rust is "

#run quantized models (usually cpu only)
#by default '7b-mistral-instruct-v0.2' weights get downloaded & loaded
cargo run --release --bin quantized -- --cpu --prompt "The best thing about coding in rust is "

#to run speciffic model see /quantiezed/src/main.rs (enum Which) for supported models 
cargo run --release --bin quantized -- --which mixtral --prompt "The best thing about coding in rust is "

# mistral (i wasn't able to run on 8Gb /dedicated --> Weights > 8gb)
cargo run --release -- --prompt 'Write helloworld code in Rust' --sample-len 150

# gemma models 

# ACCES Precondiitions > https://github.com/huggingface/candle/tree/main/candle-examples/examples/gemma
# HF auth CLI: https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started

# 1 pip install -U "huggingface_hub[cli]"
# 2 huggingface-cli login 
# 3 copy token from cli URL
cargo run --bin gemma --release -- --which code-7b-it  --prompt "fn count_primes(max_n: usize)"
# On <= 2000 series NVDA chips: 
# Error: DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found") when loading cast_u32_bf16
# See: https://github.com/huggingface/candle/issues/1911 (Only supported on RTX 3000+ chips >= 8.0 Compute capabilities)

```

**TO Resolve cuda runtime issues** see : [Error: Cuda("no cuda implementation for softmax-last-dim")#1330](https://github.com/huggingface/candle/issues/1330)<br>

```bash
#1  Add cuda feature to your candle-transformers dep (same as for the candle-core)
... "features = ["cuda"]"
 candle-transformers = { git = "https://github.com/huggingface/candle.git", version = "0.4.2", features = ["cuda"] }
#2 Run model as normal
cargo run --release --bin quantized -- --prompt "The best thing about coding in rust is "
```


#### What are Tensors? 
- 1) a 1D block of memory called Storage that holds the raw data, and 
- 2) a View over that storage that holds its shape. PyTorch Internals could be helpful here.

REF: [LLM C - Karpathy repo ](https://github.com/karpathy/llm.c/)

### Higgingface/candle model examples

[DistilBert repo](https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/distilbert)

[Mistral repo](https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/mistral)

[Quantized repo](https://github.com/huggingface/candle/blob/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/quantized)


#### Quntized mistral HF repo
[Weights for Quantized mistral [TheBloke]](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main)

### Conclusion
> Even **Quantized** models are slow when run on laptop CPU.<br/> ( 365 tokens generated: 1.23 token/s)
For RAG it seems like running 7B models locally is duable even when quantized.  <br><br>
> When you enable 'cuda' feature for transformers you get  > 421 tokens generated: 37.81 token/s (quntized model)

#### Quantization
>Weight quantization in large language models (LLMs) or any deep learning models refers to the process of reducing the precision of the model's weights from floating-point representation (e.g., 32-bit floating-point numbers) to lower bit-width representations (e.g., 16-bit, 8-bit, or even 1-bit). The primary goal of weight quantization is to reduce the memory footprint and computational requirements of the model, allowing for faster and more efficient inference on devices with limited resources, such as mobile devices or embedded systems.

Weight quantization typically involves the following steps:
1. **Weight quantization**: The model's weights are quantized from higher precision floating-point representations to lower bit-width fixed-point representations. This step usually involves finding a suitable scaling factor to maintain the dynamic range of the weights.
2. **Quantization error compensation**: This step aims to minimize the loss in accuracy caused by the quantization process. One common approach is to use a technique called "post-training quantization," where the quantized model is fine-tuned to compensate for the quantization error.
3. **Rounding and clipping**: The quantized weights are rounded to the nearest representable value within the target bit-width, potentially introducing some clipping errors in the process.

#### Simple python code that demonstrats basic RAG flow 
- [rag-from-the-ground-up-with-python](https://decoder.sh/videos/rag-from-the-ground-up-with-python-and-ollama)
- [Huggingface rag code exampels](https://huggingface.co/docs/transformers/model_doc/rag)

Good Rag posts<bre>
- [Efficient Information Retrieval with RAG Workflow](https://medium.com/@akriti.upadhyay/efficient-information-retrieval-with-rag-workflow-afdfc2619171)
- [RAG chatbot using qdrant + Gemini](https://medium.com/@akriti.upadhyay/building-real-time-financial-news-rag-chatbot-with-gemini-and-qdrant-64c0a3fbe45b)
- [Implementing RAG w HF + Langchain](https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7)

---
#### Distilbert HF

[Model doc](https://huggingface.co/docs/transformers/model_doc/distilbert)

Example python implementation

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
```
### RAG pipeline (high lvl overview)
![image (1)](https://github.com/dommyrock/doc-rag-demo/assets/32032778/9b134e26-0d5c-46b8-afb4-00aa2148e6e7)


### Windows download directory for weights/safetensors 
```bash
C:\Users\dpolzer\.cache\huggingface\hub
```

### GPU monitoring:
Check your GPU VRAM 
```bash
nvidia-smi

# or watch (loops every few sec)
nvidia-smi -l

# more details
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
#(example) NVIDIA GeForce RTX 2070 with Max-Q Design, 7.5, 552.12
```

[Cuda compiler driver nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)

<br/>
<br/>
<br/>

### Errors 

- Non matching file chunk dimensions .
If you make a mistake while splitting tokenized chunks from document to non equal (non padded) equal len.<br>
Yor matrice dimensions won't match and when you try to stack them you'll get similar error.

**ERROR: shape mismatch in cat for dim 1, shape for arg 1: [1, 84, 768] shape for arg 2: [1, 100, 768]**
```bash
    let stacked_embeddings = Tensor::stack(&embeddings_arc, 0)?;
```

- ERRORS thrown from **Qdrant** are usually because it expects one dimensional Tensor <br>
For example document chunk embeddings shape should look like<br>
> Tensor SHAPE/Dims: [12, 768]  
--> 12 chunks /embeddings of 768 dimensions (generated from 
distil bert model) <br>

> Prompt Shape/Dims: [768] --> (generated from distil bert model)

```rust
//you can either call .squeeze(0)?; //to get rid of unwanted dimension

//or get rid of multiple unwanted dimensions and only keep relevant embeddings in the Tensor 
.mean((0, 1))?; //was [1,9,768] -? keep 768
```

#### Cuda Errors
[Error: DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found") when loading cast_f32_bf16#2041](https://github.com/huggingface/candle/issues/2041)

[Capability docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

[CUDA Gpus](https://developer.nvidia.com/cuda-gpus)

