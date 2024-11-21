# from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Tokenizer

import torch
from .pretrained import T5CODE_TOKENIZER, T5TEXT_TOKENIZER, T5CODE_MODEL, T5TEXT_MODEL

def BROrMethodToVector(tokenizer, model, text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(len(input_ids[0]))
    
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids)
        last_hidden_states = encoder_outputs.last_hidden_state
    
    return last_hidden_states


if __name__ == "__main__":
    text = """def BROrMethodToVector(tokenizer, model, text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids)
        last_hidden_states = encoder_outputs.last_hidden_state
    
    return last_hidden_states"""
    vector = BROrMethodToVector(T5CODE_TOKENIZER, T5CODE_MODEL, text)
    print(vector.shape, torch.max(vector), torch.min(vector))
    vector = BROrMethodToVector(T5TEXT_TOKENIZER, T5TEXT_MODEL, text)
    print(vector.shape, torch.max(vector), torch.min(vector))
    print(vector)
