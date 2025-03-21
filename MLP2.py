import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from Tokenizer import tokenize_text, separate_tokens, split_into_sentences, split_into_paragraphs

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Using device: {device}')

#read in all the words
words = open('Dataset/Simple English/ss3.txt', 'r').read()


paragraphs = split_into_paragraphs(words)
dataset = []

for p in paragraphs:
    dataset.append(p.replace("\n", " "))
    
# build the vocabulary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['¬'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

# build the dataset

context_size = 1 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
     
    X, Y = [], []
    for w in words:

        #print(w)
        context = [0] * context_size
        for ch in w + '¬':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    print(X.shape, Y.shape)
    return X, Y

import random
random.seed(42)
random.shuffle(dataset)
n1 = int(0.8*len(dataset))
n2 = int(0.9*len(dataset))

Xtr, Ytr = build_dataset(dataset[:n1])
Xdev, Ydev = build_dataset(dataset[n1:n2])
Xte, Yte = build_dataset(dataset[n2:])

#Network

similarity_dimensions = 1
hidden_neurons = 2

g = torch.Generator(device = device).manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size, similarity_dimensions), generator=g, device = device)
W1 = torch.randn((similarity_dimensions * context_size, hidden_neurons), generator = g, device = device) * (5/3)/((similarity_dimensions * context_size)**0.5)
b1 = torch.randn(hidden_neurons, generator = g, device = device) * 0.01
W2 = torch.randn((hidden_neurons, hidden_neurons), generator = g, device = device) * 0.01
b2 = torch.randn(hidden_neurons, generator = g, device = device) * 0
W3 = torch.randn((hidden_neurons, vocab_size), generator=g, device=device) * 0.01
b3 = torch.randn(vocab_size, generator=g, device=device) * 0
parameters = [C, W1, b1, W2, b2, W3, b3]

print("Total parameters: ", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True
    
def training_loss():
    emb = C[Xtr].to(device)
    h1 = torch.tanh(emb.view(-1, similarity_dimensions * context_size) @ W1 + b1).to(device)
    h2 = torch.tanh(h1 @ W2 + b2).to(device)  # Second hidden layer
    logits = h2 @ W3 + b3  # Output layer
    loss = F.cross_entropy(logits, Ytr)
    print(f"Train loss: {loss.item()}")    

def dev_loss():
    emb = C[Xdev].to(device)
    h1 = torch.tanh(emb.view(-1, similarity_dimensions * context_size) @ W1 + b1).to(device)
    h2 = torch.tanh(h1 @ W2 + b2).to(device)  # Second hidden layer
    logits = h2 @ W3 + b3  # Output layer
    loss = F.cross_entropy(logits, Ydev)
    print(f"Dev loss: {loss.item()}")
    
start_time = time.time()
prev_time = time.time()
total_examples = 0
batch_size = 256
print_size = 5000
total_steps = 200000

for i in range(total_steps):        

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,)).to(device)
    Xb, Yb = Xtr[ix], Ytr[ix] #batch X, Y
    
    # forward pass
    emb = C[Xb].to(device)  # embed the words into vectors
    embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
    h1_preact = embcat @ W1 + b1  # first hidden layer pre-activation
    h1 = torch.tanh(h1_preact).to(device)  # first hidden layer
    h2_preact = h1 @ W2 + b2  # second hidden layer pre-activation 
    h2 = torch.tanh(h2_preact).to(device)  # second hidden layer
    logits = h2 @ W3 + b3  # output layer
    loss = F.cross_entropy(logits, Yb)  # loss function
    
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # update
    #lr = lrs[i]
    lr = 0.2 - (i  / (total_steps * 5))
    for p in parameters:
        p.data += -lr * p.grad


    if (i % print_size == 0):
        epoch_time = time.time() - start_time
        particular_time = epoch_time - prev_time
        prev_time = epoch_time
        examples = print_size * batch_size / particular_time
        total_examples += print_size * batch_size
        print(f"Steps: {i}, Lr: {lr:.3f}, time: {particular_time:.2f}, total time: {epoch_time:.2f} seconds, steps/s: {(print_size / particular_time):.2f}, examples/s: {examples:.2f}, total examples: {total_examples}, loss: {loss.item():.6f}")
    if (i % 25000 == 0):   
        dev_loss()
        training_loss()

dev_loss()
training_loss()
        
# sample from the model
g = torch.Generator(device = device).manual_seed(2147483647 + 10)
for _ in range(20):

    out = []
    context = [0] * context_size # initialize with all ...
    while True:
        emb = C[torch.tensor([context])].to(device)  # (1, context_size, d)
        h1 = torch.tanh(emb.view(1, -1) @ W1 + b1).to(device)  # First hidden layer
        h2 = torch.tanh(h1 @ W2 + b2).to(device)  # Second hidden layer
        logits = h2 @ W3 + b3  # Output layer
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix != 0:
            out.append(ix)
        if ix == 0:
            break
            
    print(''.join(itos[i] for i in out))
    
def save_model(parameters, itos, filename):
    """
    Save model parameters and character mapping to a file.
    
    Args:
        parameters: List of model parameters [C, W1, b1, W2, b2]
        itos: Dictionary mapping integers to characters
        filename: Name of the file to save the model to
    """
    C, W1, b1, W2, b2, W3, b3 = parameters
    torch.save({
        'C': C,
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3,
        'itos': itos,
        'context_size': context_size,
        'vocab_size': len(itos)
    }, filename)
    print(f"Model saved to {filename}")

def load_model_and_sample(filename, num_samples=20, seed=None):
    """
    Load model parameters and character mapping from a file and sample from the model.
    
    Args:
        filename: Name of the file containing the saved model
        num_samples: Number of words to generate
        seed: Random seed for sampling
    
    Returns:
        Dictionary containing the loaded model parameters and mappings
    """
    # Load the model
    checkpoint = torch.load(filename)
    C = checkpoint['C']
    W1 = checkpoint['W1']
    b1 = checkpoint['b1']
    W2 = checkpoint['W2']
    b2 = checkpoint['b2']
    W3 = checkpoint['W3']
    b3 = checkpoint['b3']
    itos = checkpoint['itos']
    context_size = checkpoint.get('context_size', 3)  # Default to 3 if not saved
    vocab_size = checkpoint.get('vocab_size', len(itos))
    
    device = C.device
    similarity_dimensions = C.shape[1]
    
    print(f"Model loaded from {filename}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using device: {device}")
    
    # Sample from the model
    if seed is None:
        g = torch.Generator(device=device).manual_seed(int(time.time()))
    else:
        g = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Generating {num_samples} samples:")
    for i in range(num_samples):
        out = []
        context = [0] * context_size  # initialize with start tokens
        while True:
            emb = C[torch.tensor([context], device=device)]
            h1 = torch.tanh(emb.view(1, -1) @ W1 + b1)
            h2 = torch.tanh(h1 @ W2 + b2)
            logits = h2 @ W3 + b3
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix != 0:
                out.append(ix)
            if ix == 0:
                break
                
        print(f"{i+1}. {''.join(itos[i] for i in out)}")
    
    # Return the loaded model components in case they're needed
    return {
        'C': C, 
        'W1': W1, 
        'b1': b1, 
        'W2': W2, 
        'b2': b2, 
        'itos': itos,
        'context_size': context_size,
        'vocab_size': vocab_size
    }
        
save_model([C, W1, b1, W2, b2, W3, b3], itos, "MLP2.w")