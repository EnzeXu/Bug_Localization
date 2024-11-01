from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Tokenizer

T5CODE_TOKENIZER = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
T5CODE_MODEL = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

T5TEXT_TOKENIZER = T5Tokenizer.from_pretrained("google-t5/t5-small", legacy=True)
T5TEXT_MODEL = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")