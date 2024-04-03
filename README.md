


### Build / Run

```bash
# build 
cargo b

#run
cargo run -- --prompt "Here is a test sentence"

#run release
cargo run --release -- --prompt "Here is a test sentence"

#run jina-bert in release (gpu not supported > no cuda implementation for softmax-last-dim)
cargo run --release --bin jina-bert -- --cpu --prompt "The best thing about coding in rust is "

#run quantized models (usually cpu only)
#by default '7b-mistral-instruct-v0.2' weights get downloaded & loaded
cargo run --release --bin quantized -- --cpu --prompt "The best thing about coding in rust is "

#to run speciffic model see /quantiezed/src/main.rs (enum Which) for supported models 
cargo run --release --bin quantized -- --which mixtral --prompt "The best thing about coding in rust is "

# mistral (i wasn't able to run on 8Gb /dedicated --> Weights > 8gb)
cargo run --release -- --prompt 'Write helloworld code in Rust' --sample-len 150
```

```bash
#build for cuda
cargo build --release --features cuda

#if you dont and try run on cuda build > Error: no cuda implementation for rms-norm

#even if you still build for cuda model might not support running it on cuda and you get the same error.
cargo run --release --bin quantized -- --prompt "The best thing about coding in rust is "
```


### Higgingface/candle model examples

[DistilBert repo](https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/distilbert)

[Mistral repo](https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/mistral)

[Quantized repo](https://github.com/huggingface/candle/blob/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/quantized)


#### Quntized mistral HF repo
[Weights for Quantized mistral [TheBloke]](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main)

### Conclusion
> Even **Quantized** models are slow when run on laptop CPU.<br/> ( 365 tokens generated: 1.23 token/s)
For RAG it seems like running 7B models locally is duable even when quantized.  

#### Quantization
>Weight quantization in large language models (LLMs) or any deep learning models refers to the process of reducing the precision of the model's weights from floating-point representation (e.g., 32-bit floating-point numbers) to lower bit-width representations (e.g., 16-bit, 8-bit, or even 1-bit). The primary goal of weight quantization is to reduce the memory footprint and computational requirements of the model, allowing for faster and more efficient inference on devices with limited resources, such as mobile devices or embedded systems.

Weight quantization typically involves the following steps:
1. **Weight quantization**: The model's weights are quantized from higher precision floating-point representations to lower bit-width fixed-point representations. This step usually involves finding a suitable scaling factor to maintain the dynamic range of the weights.
2. **Quantization error compensation**: This step aims to minimize the loss in accuracy caused by the quantization process. One common approach is to use a technique called "post-training quantization," where the quantized model is fine-tuned to compensate for the quantization error.
3. **Rounding and clipping**: The quantized weights are rounded to the nearest representable value within the target bit-width, potentially introducing some clipping errors in the process.

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

### Windows download directory for weights/safetensors 
```bash
C:\Users\dpolzer\.cache\huggingface\hub
```