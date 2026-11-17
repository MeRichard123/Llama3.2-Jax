from model import Llama3Model

def load_model(weights_path, context_length=512, *, quantisation=None):
    return Llama3Model.from_pretrained(weights_path, context_length, quantisation)


