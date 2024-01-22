import torch
import tiktoken
import numpy as np

from network import GPTLanguageModel
import constants

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # choose correct embeddings, must be the same like embeddings used for training
    embeddings = 'gpt'
    assert embeddings == 'gpt' or embeddings == 'character'

    if embeddings == 'character':
        stoi = np.load('../data/outputs/stoi.npy', allow_pickle=True).item()
        itos = np.load('../data/outputs/itos.npy', allow_pickle=True).item()
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda x: "".join([itos[i] for i in x])

    # initialize the model, based on embedding type
    vocab_size = len(stoi.keys()) if embeddings == 'character' else constants.GPT_VOCAB_SIZE
    model = GPTLanguageModel(vocab_size)
    model.to(device)

    # load parameters from a previously saved checkpoint
    checkpoint = torch.load(f'../data/outputs/model_gpt_1900_epochs.pt', map_location=device)
    model.load_state_dict(checkpoint)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    if embeddings == 'character':
        print(decode(model.generate(context, max_new_tokens=100, device=device)[0].tolist()))
    else:
        encoder = tiktoken.encoding_for_model("gpt2")
        print(encoder.decode(model.generate(context, max_new_tokens=100, device=device)[0].tolist()))