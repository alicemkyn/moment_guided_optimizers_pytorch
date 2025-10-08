import torch, torch.nn as nn, torch.nn.functional as F
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from optimizers import AdamCustom, RMSPropCustom, AdaGradCustom, AdaMaxCustom
import matplotlib.pyplot as plt

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter: yield tokenizer(text)

def collate_batch(batch):
    labels, texts = zip(*batch)
    labels = torch.tensor([1 if lbl=='pos' else 0 for lbl in labels])
    text_lens = [len(tkns) for tkns in texts]
    max_len = max(text_lens)
    padded = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i,tkns in enumerate(texts):
        padded[i, :len(tkns)] = torch.tensor(vocab(tkns))
    return padded, labels

class TextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        x = self.embed(x).mean(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_optimizer(opt_class, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter = IMDB(split="train")
    global vocab
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    train_iter = IMDB(split="train")
    data = [(label, tokenizer(text)) for (label, text) in list(train_iter)[:2000]]
    loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=collate_batch)
    model = TextModel(len(vocab)).to(device)
    optimizer = opt_class(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(2):
        total = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total/len(loader)
        print(f"[{name}] Epoch {epoch+1}: {avg:.4f}")
        losses.append(avg)
    plt.plot(losses, label=name)
    return losses

if __name__ == "__main__":
    for opt in [AdamCustom, RMSPropCustom, AdaGradCustom, AdaMaxCustom]:
        train_optimizer(opt, opt.__name__)
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("IMDB Optimizer Comparison")
    plt.savefig("results/3a.png")
