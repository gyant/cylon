# cylon

### LLama Prompts

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

In one word, what color is the sky?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Without this prompt structure the model goes nuts and never outputs an EOS token.

