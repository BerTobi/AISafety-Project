import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import os

device = torch.device("cpu")

# reading the data
# Define the folder path containing your text files
folder_path = 'Dataset/Español'  # Change this to your folder path

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

words = all_words

uwords = set(all_words)
size = len(uwords)

vocab_limit = 4000

if len(uwords) > vocab_limit:
    print(f"Very large vocabulary detected: {size} words. Consider limiting vocabulary size.")
    from collections import Counter
    word_counts = Counter(words)
    top_words = [word for word, _ in word_counts.most_common(vocab_limit)]
    uwords = set(top_words)
    print(f'Limited vocabulary to top {vocab_limit} words')

wtoi = {s:i for i,s in enumerate(uwords)}
itow = {i:s for s,i in wtoi.items()}

size = len(uwords)

N = torch.zeros(size, size, dtype = torch.int8, device = device)

# getting the Bigrams
bigrams = 0
for w1, w2 in zip(words, words[1:]):
    if w1 in uwords and w2 in uwords:
        ix1 = wtoi[w1]
        ix2 = wtoi[w2]

        if ((N[ix1, ix2] < 255).item()):
            N[ix1, ix2] += 1
            bigrams += 1
            if (bigrams % 10000 == 0):
                print(f"There are {bigrams} bigrams")

P = N / N.sum(dim = 1, keepdim = True)

def count_loss(input_list, verbose = False):
    log_likelihood = 0.0
    n = 0
    for w1, w2 in zip(words, words[1:]):
        if w1 in uwords and w2 in uwords:
            ix1 = wtoi[w1]
            ix2 = wtoi[w2]

            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1

    # higher the log likelihood (closer to 0) is better
    print(f"log Likelihood: {log_likelihood}")

    # but in loss function lower is better, so we negate it
    nll = -log_likelihood
    print(f"Negative log likelihood: {nll}")

    # normalize it
    print(f"Normalized Negative log Likelihood: {(nll / n)}") # we need to minimize this
    
    
print("Training Loss")
count_loss(words)

g = torch.Generator().manual_seed(2147483647)

# Sampling
sample = ""
for i in range(10):

    ix = 0
    for j in range(20):
        p = P[ix]

        ix = torch.multinomial(p, 1, replacement=True).item()
        sample += " " + itow[ix]


    
print(sample)
print("Sampled words Loss")
count_loss(sample)