from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class T5Rewriter:
    def __init__(self, model_name="t5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def rewrite(self, question: str) -> str:
        input_text = f"paraphrase: {question}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
