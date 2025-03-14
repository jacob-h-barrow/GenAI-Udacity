"""
  Errors:
    - top_next_token
    - test_generation: produces the same sentence 3x. Probably due to no gradient saves.
  Warnings:
    - test_generation: The attention mask is not set and cannot be inferred from input because pad token is same as eos token.
"""

import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any

class Tokenizer:
    @staticmethod
    def display_tokenization(data_inputs, columns=['id', 'token']) -> pd.DataFrame | bool:
        if not data_inputs:
            return False
        return pd.DataFrame(data_inputs, columns=columns)

    def __init__(self, model: str = 'gpt2'):
        self.__model_name = model

        try:
            self.__tokenizer = AutoTokenizer.from_pretrained(model)
            self.__model = AutoModelForCausalLM.from_pretrained(model)
        except Exception as e:
            raise Exception(f"{model} isn't compatible: {e}")

    @property
    def model_name(self):
        return self.__model_name

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, new_model: str) -> bool:
        return self.change_model(new_model)

    @tokenizer.setter
    def tokenizer(self, new_model: str) -> bool:
        return self.change_model(new_model)

    def change_model(self, new_model: str) -> bool:
        try:
            self.__tokenizer = AutoTokenizer.from_pretrained(new_model)
            self.__model = AutoModelForCausalLM.from_pretrained(new_model)
            self.__model_name = new_model
            return True
        except Exception as e:
            print(f"Failed to load model {new_model}: {e}")
            return False

    def test(self, text, probabilities_ptr: Any = None, _return: bool = False, option: str = 'inputs', columns=['id', 'token']):
        inputs = self.new_input(text)

        if not inputs:
            print("Tokenization failed!")
            return False

        with torch.no_grad():
            tokenized_inputs = self.__tokenizer(text, return_tensors="pt")
            logits = self.__model(**tokenized_inputs).logits[:, -1, :]

            try:
                probabilities = probabilities_ptr(logits[0], dim=-1) if probabilities_ptr else None
            except:
                probabilities = None

        match option:
            case 'inputs':
                fxn_ptr = self.new_input
                data = fxn_ptr(text)
            case 'next_tokens':
                fxn_ptr = self.top_next_token
                data = fxn_ptr(probabilities)
            case 'test_next_token':
                if probabilities is None:
                    print("Error: probabilities are None in test()")
                    return False
                fxn_ptr = self.test_next_token
                data = fxn_ptr(probabilities)
            case 'generation':
                fxn_ptr = self.test_generation
                data = fxn_ptr(text)
            case _:
                return False

        if data:
            return data if _return else print(self.display_tokenization(data, columns=columns))
        else:
            print("Operation failed!")

    def test_generation(self, text):
        try:
            inputs = self.__tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            output_ids = self.__model.generate(input_ids, max_length=100, pad_token_id=self.__tokenizer.eos_token_id)
            return self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in text generation: {e}")
            return False

    def new_input(self, text: str, return_tensors: str = 'pt'):
        try:
            inputs = self.__tokenizer(text, return_tensors=return_tensors)
            return [(_id, self.__tokenizer.decode(_id)) for _id in inputs['input_ids'][0]]
        except Exception as e:
            print(f"Error in new_input: {e}")
            return False

    def top_next_token(self, probabilities, top_n: int = 5):
        try:
            return sorted(
                [(_id, self.__tokenizer.decode(_id), p.item()) for _id, p in enumerate(probabilities)],
                key=lambda x: x[2],
                reverse=True
            )[:top_n]
        except Exception as e:
            print(f"Error in top_next_token: {e}")
            return False

    def test_next_token(self, probabilities):
        try:
            next_token_id = torch.argmax(probabilities).item()
            return self.__tokenizer.decode(next_token_id)
        except Exception as e:
            print(f"Error in test_next_token: {e}")
            return False


if __name__ == "__main__":
    token_handler = Tokenizer()
    text = "Rust is the best systems programming language"
    softmax_ptr = torch.nn.functional.softmax

    token_handler.test(text, columns=['id', 'token'])

    token_handler.test(text, probabilities_ptr=lambda logits: softmax_ptr(logits, dim=-1), columns=['id', 'token', 'p'], option='next_tokens')

    next_token = token_handler.test_next_token(softmax_ptr(torch.randn(50257), dim=-1))  # Fixed softmax call
    print(f"Next token: {next_token}")

    generated_text = token_handler.test(text, _return=True, option='generation')
    print(f"Generated text: {generated_text}")
