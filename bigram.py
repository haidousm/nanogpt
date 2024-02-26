import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# hyperparameters
batch_size = 32 # number of sequences in a batch
context_length = 8 # length of a sequence
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')
n_embd = 32
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

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

  def forward(self, x):
    _, time, channels = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    attn = (q @ k.transpose(-2, -1)) / (channels ** 0.5) # weights
    attn = attn.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)

    out = attn @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(n_embd, n_embd) # project the concatenated heads for residual connection

  def forward(self, x):
    x = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.proj(x)

class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd) # for residual connection,
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication (self-attention) + computation (feed-forward) """
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa_heads = MultiHeadAttention(n_head, head_size) # self-attention heads
    self.ff = FeedForward() # feed-forward, simple 1-layer MLP (multi-layer perceptron)
    self.ln1 = nn.LayerNorm(n_embd) # layer normalization so that the input to the block has mean 0 and std 1 (helps with training)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x)) # x + self-attention heads (residual connection)
    x = x + self.ff(self.ln2(x)) # x + feed-forward (residual connection)
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(context_length, n_embd)

    self.transformer_blocks = nn.Sequential(
      Block(n_embd, n_head=4),
      Block(n_embd, n_head=4),
      Block(n_embd, n_head=4),
      nn.LayerNorm(n_embd)
    )
    self.lm_head = nn.Linear(n_embd, vocab_size) # language model head

  def forward(self, x, targets=None):
    batch, time = x.shape

    tok_embd = self.token_embedding_table(x)
    pos_embd = self.position_embedding_table(torch.arange(time, device=device))
    x = tok_embd + pos_embd

    x = self.transformer_blocks(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      batch, time, channels = logits.shape
      logits = logits.view(batch*time, channels)
      targets = targets.view(batch * time)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, x, max_new_tokens):
    for _ in range(max_new_tokens):
      x_cond = x[:, -context_length:]

      logits, loss = self(x_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, 1)
      x = torch.cat([x, next_token.view(-1, 1)], dim=1)
    return x



model = BigramLanguageModel()
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
