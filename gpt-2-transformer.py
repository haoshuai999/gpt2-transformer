

import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary), Load pre-trained model (weights)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').eval()

def predict(text, size):

    # Encode a text inputs

    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor, past = torch.tensor([indexed_tokens]), None

    for i in range(size):
        tokens_tensor, past = model(tokens_tensor, past=past)
        tokens_tensor = torch.multinomial(F.softmax(tokens_tensor[:, -1], dim=1),1)
        indexed_tokens.append(tokens_tensor.item())

    return tokenizer.decode(indexed_tokens)


    #     # Predict all tokens
    #     with torch.no_grad():
    #         outputs = model(tokens_tensor)
    #         predictions = outputs[0]
    #         print(predictions[-1, :].shape)
    #
    #     # get the predicted next sub-word (in our case, the word 'man')
    #     predicted_index = torch.argmax(predictions[0, -1, :]).item()
    #     tokens_tensor = tokenizer.decode(indexed_tokens + [predicted_index])
    # return tokens_tensor

if __name__ == '__main__':
    test = 'Trump said he will whole-heartedly support his impeachment. This is unprecedented.'
    predict_text = predict(test, 100).strip()
    print(predict_text)