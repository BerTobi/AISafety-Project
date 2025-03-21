import math
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

words = open('Dataset/Test 3.txt', 'r').read().split()
uwords = set(words)

wtoi = {s:i for i,s in enumerate(uwords)}
itow = {i:s for s,i in wtoi.items()}

# create the dataset
xs, ys = [], []
for w1, w2, w3 in zip(words, words[1:], words[2:]):
    ix1 = wtoi[w1]
    ix2 = wtoi[w2]
    ix3 = wtoi[w3]
    xs.append((ix1, ix2))
    ys.append(ix3)
xs = torch.tensor(xs).to(device)
ys = torch.tensor(ys).to(device)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(10022004)
W = torch.randn((len(uwords)*2, len(uwords)), requires_grad=True, device = device)

# gradient descent
for k in range(3000):

    # forward pass
    xenc = F.one_hot(xs, num_classes=len(uwords)).float().to(device) # input to the network: one-hot encoding
    logits = xenc.view(-1, len(uwords)*2) @ W # predict log-counts
    #counts = logits.exp() # counts, equivalent to N
    #probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
    #loss = -probs[torch.arange(num), ys].log().mean()
    loss = torch.nn.functional.cross_entropy(logits, ys)
    if (k % 100 == 0):
        print("step: ", k, "loss: ", loss.item())

    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()

    # update
    with torch.no_grad():
        W -= 10 * W.grad
    
    # finally, sample from the 'neural net' model
g = torch.Generator(device= device).manual_seed(2147483647)

for i in range(5):

    out = ""
    ix1, ix2 = 0, 0
    for j in range(20):

        xenc = F.one_hot(torch.tensor((ix1, ix2)), num_classes=len(uwords)).float().to(device)
        logits = xenc.view(-1, len(uwords)*2) @ W # predict log-counts
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True) # probabilities for next word

        #p_cpu = p.cpu() if device.type == 'cuda' else p
        ix1 = ix2
        ix2 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out += " " + itow[ix2]
    print(''.join(out))