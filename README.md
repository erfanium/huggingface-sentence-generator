
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")


class MyGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        with torch.no_grad():
            self._masking = torch.zeros(
                1, 1, config.vocab_size, device=self.device, dtype=self.dtype
            )
            self._masking[0, 0, tokenizer.eos_token_id] = -2

    def __call__(self, *inputs, **kwargs):
        result = super().__call__(*inputs, **kwargs)
        result.logits += self._masking
        return result
```
