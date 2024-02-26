import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# hyperparameters
batch_size = 32 # number of sequences in a batch
context_length = 8 # length of a sequence
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')
# -------------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/input.txt
with open('data/input.txt', 'r') as file:
    text = file.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# ---- tokenizer ----
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# ---- train/val split ----
data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(0, len(data) - context_length, (batch_size,))
  x = torch.stack([data[idx:idx+context_length] for idx in ix])
  y = torch.stack([data[idx+1:idx+context_length+1] for idx in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      xb, yb = get_batch(split)
      logits, loss = model(xb, yb)
      losses[k] = loss.item()
    out[split] = losses.mean().item()
  model.train()
  return out

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, x, targets=None):

    logits = self.token_embedding_table(x)

    loss = None
    if targets is not None:
      batch, time, channels = logits.shape
      logits = logits.view(batch*time, channels)
      targets = targets.view(batch * time)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, x, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(x)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, 1)
      x = torch.cat([x, next_token.view(-1, 1)], dim=1)
    return x



model = BigramLanguageModel(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = time.time()
for _iter in range(max_iters):
  if _iter % eval_interval == 0:
    losses = estimate_loss()
    print(f'iter {_iter}, train loss: {losses["train"]}, val loss: {losses["val"]}')
  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

end = time.time()
print('Training time:', end - start)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
