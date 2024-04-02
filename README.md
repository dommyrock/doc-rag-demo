


### Build / Run

```bash
# build 
cargo b

#run
cargo run -- --prompt "Here is a test sentence"

#run release
cargo run --release -- --prompt "Here is a test sentence"

# mistral (i wasn't able to run on 8Gb /dedicated )
cargo run --release -- --prompt 'Write helloworld code in Rust' --sample-len 150
```



### Higgingface model examples

[DistilBert repo](https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/distilbert)

[Mistral repo](https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/mistral)

[Quantized repo](https://github.com/huggingface/candle/blob/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/quantized)


#### Quntized mistral HF repo
[Weights for Quantized mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main)

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
