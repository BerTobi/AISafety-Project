import math
import torch
import torch.nn.functional as F
import os
import glob
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the folder path containing your text files
folder_path = 'Dataset/Test English'  # Change this to your folder path

# Get all text files in the folder
file_paths = glob.glob(os.path.join(folder_path, '*.txt'))
print(f'Found {len(file_paths)} text files in {folder_path}')

# Read and combine all words from all files
all_words = []
for file_path in file_paths:
    try:
        print(f'Processing file: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            words = file.read().split()
            all_words.extend(words)
    except Exception as e:
        print(f'Error processing {file_path}: {str(e)}')

print(f'Total words read: {len(all_words)}')

uwords = set(all_words)
size = len(uwords)

vocab_limit = 2000

if len(uwords) > vocab_limit:
    print(f"Very large vocabulary detected: {size} words. Consider limiting vocabulary size.")
    from collections import Counter
    word_counts = Counter(words)
    top_words = [word for word, _ in word_counts.most_common(vocab_limit)]
    uwords = set(top_words)
    print(f'Limited vocabulary to top {vocab_limit} words')
    
wtoi = {s:i for i,s in enumerate(uwords)}
itow = {i:s for s,i in wtoi.items()}

# create the dataset
xs, ys = [], []
for w1, w2 in zip(words, words[1:]):
    if w1 in uwords and w2 in uwords:
        ix1 = wtoi[w1]
        ix2 = wtoi[w2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs).to(device)
ys = torch.tensor(ys).to(device)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(10022004)
W = torch.randn((len(uwords), len(uwords)), requires_grad=True, device = device)

start_time = time.time()
prev_time = start_time

# gradient descent
for k in range(5000):

    # forward pass
    xenc = F.one_hot(xs, num_classes=len(uwords)).to(torch.float32).to(device) # input to the network: one-hot encoding
    logits = xenc @ W # predict log-counts
    #counts = logits.exp() # counts, equivalent to N
    #probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
    #loss = -probs[torch.arange(num), ys].log().mean()
    loss = torch.nn.functional.cross_entropy(logits, ys)
    if (k % 25 == 0):
        epoch_time = time.time() - start_time
        particular_time = epoch_time - prev_time
        prev_time = epoch_time
        print(f"step: {k}, loss: {loss.item()}, time: {particular_time:.2f}, total time: {epoch_time:.2f} seconds")

    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()

    # update
    with torch.no_grad():
        W -= 50 * W.grad
    
    # finally, sample from the 'neural net' model
g = torch.Generator(device= device).manual_seed(2147483647)

for i in range(5):

    out = ""
    ix = 0
    for j in range(20):

        xenc = F.one_hot(torch.tensor([ix]), num_classes=len(uwords)).to(torch.float32).to(device)
        logits = xenc @ W # predict log-counts
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True) # probabilities for next word

        #p_cpu = p.cpu() if device.type == 'cuda' else p
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out += " " + itow[ix]
    print(''.join(out))