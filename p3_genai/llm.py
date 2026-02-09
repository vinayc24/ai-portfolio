"""
Docstring for p3_genai.llm

Defines a lightweight local LLM wrapper
Keeps model loading separate from RAG logic


"""

from transformers import pipeline

class LocalLLM:
    """
    Wrapper around a small text-generation model.
    Designed to be CPU friendly
    """
    def __init__(self):
        """
        Docstring for __init__
        Initialize the local Language model
        
        :param self: Description
        """
        self.generator = pipeline(
            task="text-generation",
            model="distilgpt2",
            do_sample=False
        )

    def generate(self, prompt:str) -> str:
        """
        Generates an answer given a prompt
        Args prompt(str): Prompt contaning context +question
        Returns:
            str:Generated answer
        """
        
        output = self.generator(
            prompt,
            max_new_tokens=100,   # ✅ generate only new tokens
            truncation=True
        )
        return output[0]["generated_text"]