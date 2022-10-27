import torch
from torch import nn
from IPython import embed
import math
import einops

class SelfAttention(nn.Module):
  def __init__(self, dim, heads=8):
    super().__init__()
    self.dim, self.heads = dim, heads

    self.innerprod = nn.Parameter(torch.empty(heads, dim, dim))
    self.tovalue = nn.Parameter(torch.empty(heads, dim, dim))
    self.unifyheads = nn.Parameter(torch.empty(heads, dim, dim))
    with torch.no_grad():
        nn.init.xavier_uniform_(self.innerprod)
        nn.init.xavier_uniform_(self.tovalue)
        nn.init.xavier_uniform_(self.unifyheads)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    b, c, l = x.size()
    h = self.heads

    # Compute inner products between keys and queries
    weights = torch.einsum('bki,hkq,bqj->bhij', x, self.innerprod, x) / math.sqrt(c)
    weights = self.softmax(weights)

    # Transform input to value
    x = torch.einsum('bvi,htv->bhti', x, self.tovalue)

    # Compute the weighted sum of the value
    x = torch.einsum('bhij,bhti->bhtj', weights, x)

    # Unify the individual heads
    x = torch.einsum('bhtl,hct->bcl', x, self.unifyheads)

    return x

class SelfAttentionSparse(nn.Module):
  def __init__(self, dim, embedding_dim=None, heads=8, value_identity=True):
    super().__init__()
    self.dim, self.heads = dim, heads

    if embedding_dim is None:
        embedding_dim = dim
    self.embedding_dim = embedding_dim

    self.value_identity = value_identity

    self.tokey = nn.Parameter(torch.empty(heads, embedding_dim, dim))
    self.toquery = nn.Parameter(torch.empty(heads, embedding_dim, dim))

    if not value_identity:
        self.tovalue = nn.Parameter(torch.empty(heads, dim, dim))

    self.unifyheads = nn.Parameter(torch.empty(heads, dim, dim))

    with torch.no_grad():
        nn.init.xavier_uniform_(self.tokey)
        nn.init.xavier_uniform_(self.toquery)
        nn.init.xavier_uniform_(self.unifyheads)
        if not value_identity:
            nn.init.xavier_uniform_(self.tovalue)

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    b, c, l, h = *x.size(), self.heads

    # Compute inner products between keys and queries
    key = torch.einsum('bki,hek->bhei', x, self.tokey)
    query = torch.einsum('heq,bqj->bhej', self.toquery, x)
    weights = torch.einsum('bhei, bhej -> bhij', key, query) / math.sqrt(c)

    #weights = torch.einsum('bki,hek,heq,bqj->bhij', x, self.tokey, self.toquery, x) / math.sqrt(c)

    weights = self.softmax(weights)

    if self.value_identity:
        x = einops.repeat(x, 'b v i -> b h v i', h = h)
    else:
        # Transform input to value
        x = torch.einsum('bvi,htv->bhti', x, self.tovalue)

    # Compute the weighted sum of the value
    x = torch.einsum('bhij,bhti->bhtj', weights, x)

    # Unify the individual heads
    x = torch.einsum('bhtl,hct->bcl', x, self.unifyheads)

    return x
