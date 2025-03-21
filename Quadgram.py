import math
import torch
import torch.nn.functional as F
import time
import random
from Tokenizer import tokenize_text, separate_tokens, split_into_sentences, split_into_paragraphs

words = open('Dataset/Simple English/pg11.txt', 'r').read()

paragraphs = split_into_paragraphs(words)
dataset = []

for p in paragraphs:
    dataset.append(p.replace("\n", " "))

c = len(dataset)

random.seed(42)
random.shuffle(dataset)

train_set = dataset[:math.floor(c*0.8)]
dev_set = dataset[math.floor(c*0.8):math.floor(c*0.9)]
test_set = dataset[math.floor(c*0.9):]

chars = sorted(list(set(''.join(dataset))))
stoi = {s:i + 1 for i,s in enumerate(chars)}
stoi['¬'] = 0
itos = {i:s for s,i in stoi.items()}

vocab_size = len(stoi)

# create the dataset

def build_dataset(dataset):
    X, Y = [], []
    for w in dataset:
        chs = ['¬'] + ['¬'] + ['¬'] + list (w) + ['¬'] + ['¬'] + ['¬']
        for ch1, ch2, ch3, ch4 in zip(chs, chs[1:], chs[2:], chs[3:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            ix4 = stoi[ch4]
            X.append((ix1, ix2, ix3))
            Y.append(ix4)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

Xtr, Ytr = build_dataset(train_set)
Xdev, Ydev = build_dataset(dev_set)
Xte, Yte = build_dataset(test_set)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size * 3, len(stoi)), generator=g, requires_grad=True)

start_time = time.time()
prev_time = time.time()
total_examples = 0
batch_size = 256

def training_loss():

    # forward pass
    xenc = F.one_hot(Xtr, num_classes=len(stoi)).float() # input to the network: one-hot encoding
    logits = xenc.view(-1, vocab_size * 3) @ W # predict log-counts
    loss = torch.nn.functional.cross_entropy(logits, Ytr)
    print("Training loss: ", loss.item())

def dev_loss():

    # forward pass
    xenc = F.one_hot(Xdev, num_classes=len(stoi)).float() # input to the network: one-hot encoding
    logits = xenc.view(-1, vocab_size * 3) @ W # predict log-counts
    loss = torch.nn.functional.cross_entropy(logits, Ydev)
    print("Dev loss: ", loss.item())

# gradient descent
for k in range(200000):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    # forward pass
    xenc = F.one_hot(Xtr[ix], num_classes=len(stoi)).float() # input to the network: one-hot encoding
    logits = xenc.view(-1, vocab_size * 3) @ W # predict log-counts
    loss = torch.nn.functional.cross_entropy(logits, Ytr[ix])
    if (k % 2500 == 0):
        print("step: ", k, "loss: ", loss.item())

    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()

    # update
    W.data += -0.1 * W.grad

training_loss()
dev_loss()

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(10):

    out = []
    ix1, ix2, ix3 = 0, 0, 0
    while True:

        xenc = F.one_hot(torch.tensor((ix1, ix2, ix3)), num_classes=len(stoi)).float()
        logits = xenc.view(-1, vocab_size * 3) @ W # predict log-counts
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True) # probabilities for next character

        ix1 = ix2
        ix2 = ix3
        ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix3 != 0:
            out.append(itos[ix3])
        if ix3 == 0:
            break
    print(''.join(out))
