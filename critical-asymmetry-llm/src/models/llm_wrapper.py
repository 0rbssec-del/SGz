import torch


class LLMWrapper:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, text, return_tensors="pt"):
        return self.tokenizer(text, return_tensors=return_tensors)

    def __call__(self, **inputs):
        return self.model(**inputs)

    def logits(self, prompt, system=None):
        if system:
            prompt = f"{system}\n\n{prompt}"
        inputs = self.tokenize(prompt).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1, :]

    def sample(self, logits, T=1.0, p=0.9):
        probs = torch.softmax(logits / T, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative <= p
        mask[0] = True
        sorted_probs = sorted_probs * mask
        sorted_probs = sorted_probs / sorted_probs.sum()
        idx = torch.multinomial(sorted_probs, 1)
        return sorted_idx[idx].item()
