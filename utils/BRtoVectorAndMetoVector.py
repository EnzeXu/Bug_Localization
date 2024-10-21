import transforms
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch

T5Code_tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
T5Code_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

def BROrMethodToVector(tokenizer, model, text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state

    return last_hidden_states