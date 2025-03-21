import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Using device: {device}')

#read in all the words
words = open('Dataset/Simple English/ss3.txt', 'r').read().splitlines()

# build the vocabulary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['¬'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

# build the dataset

context_size = 2 # context length: how many characters do we take to predict the next one?

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
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# The dataset was built the same way as in the previous version.

# The classes we create here are the same API as nn.Module in PyTorch

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

n_embd = 2 # the dimensionality of the character embedding vectors
n_hidden = 8 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, n_embd),            generator=g)
#layers = [
#  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
#]
layers = [
  Linear(n_embd * context_size, n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size),
]

with torch.no_grad():
  # last layer: make less confident
  #layers[-1].gamma *= 0.1
  #layers[-1].weight *= 0.1
  # all other layers: apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  x = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)
  print(split, loss.item())

# put layers into eval mode
for layer in layers:
  layer.training = False

# same optimization as last time
max_steps = 50000
batch_size = 32
lossi = []
ud = []
start_time = time.time()
prev_time = time.time()
print_size = 5000
total_examples = 0

for i in range(max_steps):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  # forward pass
  emb = C[Xb] # embed the characters into vectors
  x = emb.view(emb.shape[0], -1) # concatenate the vectors
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb) # loss function
  
  # backward pass
  for layer in layers:
    layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % print_size == 0: # print every once in a while
    epoch_time = time.time() - start_time
    particular_time = epoch_time - prev_time
    prev_time = epoch_time
    examples = print_size * batch_size / particular_time
    total_examples += print_size * batch_size
    print(f"Steps: {i}, Lr: {lr:.3f}, time: {particular_time:.2f}, total time: {epoch_time:.2f} seconds, steps/s: {(print_size / particular_time):.2f}, examples/s: {examples:.2f}, total examples: {total_examples}, loss: {loss.item():.6f}")
  if i % (print_size * 5) == 0:
    split_loss('train')
    split_loss('val')  
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])
    
split_loss('train')
split_loss('val')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * context_size # initialize with all ...
    while True:
      # forward pass the neural net
      emb = C[torch.tensor([context])] # (1,block_size,n_embd)
      x = emb.view(emb.shape[0], -1) # concatenate the vectors
      for layer in layers:
        x = layer(x)
      logits = x
      probs = F.softmax(logits, dim=1)
      # sample from the distribution
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      # shift the context window and track the samples
      context = context[1:] + [ix]
      if ix != 0:
        out.append(ix)
      # if we sample the special '.' token, break
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # decode and print the generated word
    
def save_model(model_params, filename):
    """
    Save model parameters, vocabulary, and context size to a file.
    
    Args:
        model_params: List of model parameters (weights and biases)
        filename: Name of the file to save the model to
    """
    # Create a dictionary to store all parameters
    model_state = {
        'C': model_params[0],  # Character embeddings
        'layers': [],
        'vocab': {
            'stoi': stoi,
            'itos': itos,
            'vocab_size': vocab_size
        },
        'context_size': context_size
    }
    
    # Store parameters for each layer
    layer_params = model_params[1:]
    layer_idx = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, Linear):
            model_state['layers'].append({
                'type': 'Linear',
                'weight': layer_params[layer_idx],
                'bias': layer_params[layer_idx + 1] if layer.bias is not None else None
            })
            layer_idx += 1 if layer.bias is None else 2
        elif isinstance(layer, BatchNorm1d):
            model_state['layers'].append({
                'type': 'BatchNorm1d',
                'gamma': layer_params[layer_idx],
                'beta': layer_params[layer_idx + 1],
                'running_mean': layer.running_mean,
                'running_var': layer.running_var
            })
            layer_idx += 2
        elif isinstance(layer, Tanh):
            model_state['layers'].append({
                'type': 'Tanh'
            })
    
    # Save the model state to a file
    torch.save(model_state, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load model parameters, vocabulary, and context size from a file.
    
    Args:
        filename: Name of the file to load the model from
        
    Returns:
        List of model parameters (weights and biases)
    """
    # Load the model state from the file
    model_state = torch.load(filename)
    
    # Extract the character embeddings
    C = model_state['C']
    
    # Load vocabulary and context size
    global stoi, itos, vocab_size, context_size
    stoi = model_state['vocab']['stoi']
    itos = model_state['vocab']['itos']
    vocab_size = model_state['vocab']['vocab_size']
    context_size = model_state['context_size']
    
    print(f"Loaded vocabulary size: {vocab_size}")
    print(f"Loaded context size: {context_size}")
    
    # Create new layers with the loaded parameters
    new_layers = []
    for i, layer_info in enumerate(model_state['layers']):
        if layer_info['type'] == 'Linear':
            layer = Linear(layer_info['weight'].shape[0], layer_info['weight'].shape[1], 
                          bias=layer_info['bias'] is not None)
            layer.weight = layer_info['weight']
            if layer.bias is not None:
                layer.bias = layer_info['bias']
            new_layers.append(layer)
        elif layer_info['type'] == 'BatchNorm1d':
            dim = layer_info['gamma'].shape[0]
            layer = BatchNorm1d(dim)
            layer.gamma = layer_info['gamma']
            layer.beta = layer_info['beta']
            layer.running_mean = layer_info['running_mean']
            layer.running_var = layer_info['running_var']
            new_layers.append(layer)
        elif layer_info['type'] == 'Tanh':
            new_layers.append(Tanh())
    
    # Replace the global layers with the new ones
    global layers
    layers = new_layers
    
    # Return the list of parameters
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    for p in parameters:
        p.requires_grad = True
    
    print(f"Model loaded from {filename}")
    return parameters

# Example usage:
# After training:
# save_model(parameters, 'mlp_model.pt')

# To load the model:
# parameters = load_model('mlp_model.pt')

# Function to generate words using the loaded model
def generate_words(num_words=20, seed=None):
    """
    Generate words using the loaded model.
    
    Args:
        num_words: Number of words to generate
        seed: Random seed for reproducibility
    """
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    else:
        g = torch.Generator().manual_seed(2147483647 + 10)
    
    # Put layers into eval mode
    for layer in layers:
        if hasattr(layer, 'training'):
            layer.training = False
    
    for _ in range(num_words):
        out = []
        context = [0] * context_size  # initialize with all zeros
        while True:
            # forward pass the neural net
            emb = C[torch.tensor([context])]  # (1,context_size,n_embd)
            x = emb.view(emb.shape[0], -1)  # concatenate the vectors
            for layer in layers:
                x = layer(x)
            logits = x
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            # shift the context window and track the samples
            context = context[1:] + [ix]
            if ix != 0:
                out.append(ix)
            # if we sample the special '.' token, break
            if ix == 0:
                break
        
        print(''.join(itos[i] for i in out))  # decode and print the generated word
    
parameters = [C] + [p for layer in layers for p in layer.parameters()]
save_model(parameters, 'MLP2.w')