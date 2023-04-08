import torch
from torch import nn
from IPython import embed
import math
import einops

class SelfAttention(nn.Module):
  def __init__(self, dim, embedding_dim=None, heads=8):
    super().__init__()
    self.dim, self.heads = dim, heads

    if embedding_dim is None:
        embedding_dim = math.ceil(dim/heads)

    self.innerprod = nn.Parameter(torch.empty(heads, dim, dim))
    self.tovalue = nn.Parameter(torch.empty(heads, embedding_dim, dim))
    self.tovalue_bias = nn.Parameter(torch.empty(1, heads, embedding_dim, 1))
    self.unifyheads = nn.Parameter(torch.empty(heads, dim, embedding_dim))
    self.unifyheads_bias = nn.Parameter(torch.empty(1, dim, 1))

    l = math.sqrt(1/dim)

    with torch.no_grad():
        nn.init.xavier_uniform_(self.innerprod)
        nn.init.xavier_uniform_(self.tovalue)
        nn.init.xavier_uniform_(self.unifyheads)
        nn.init.uniform_(self.unifyheads_bias, a=-l, b=l)
        nn.init.zeros_(self.tovalue_bias)

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
    x = torch.einsum('bhij,bhti->bhtj', weights, x) + self.tovalue_bias

    # Unify the individual heads
    x = torch.einsum('bhtl,hct->bcl', x, self.unifyheads) + self.unifyheads_bias

    return x

class SelfAttentionSparse(nn.Module):
  def __init__(self, dim, inner_dim=None, embedding_dim=None, heads=8):#, value_identity=True):
    super().__init__()
    self.dim, self.heads = dim, heads

    if embedding_dim is None:
        embedding_dim = math.ceil(dim/heads)
    if inner_dim is None:
        inner_dim = embedding_dim

    self.embedding_dim = embedding_dim
    self.inner_dim = inner_dim

    self.tokey = nn.Parameter(torch.empty(heads, inner_dim, dim))
    self.tokey_bias = nn.Parameter(torch.empty(1, heads, inner_dim, 1))

    self.toquery = nn.Parameter(torch.empty(heads, inner_dim, dim))
    self.toquery_bias = nn.Parameter(torch.empty(1, heads, inner_dim, 1))

    self.tovalue = nn.Parameter(torch.empty(heads, embedding_dim, dim))
    self.tovalue_bias = nn.Parameter(torch.empty(1, heads, embedding_dim, 1))

    self.unifyheads = nn.Parameter(torch.empty(heads, dim, embedding_dim))
    self.unifyheads_bias = nn.Parameter(torch.empty(1, dim, 1))

    l = math.sqrt(1/dim)

    with torch.no_grad():
        nn.init.xavier_uniform_(self.tokey)
        nn.init.xavier_uniform_(self.toquery)
        nn.init.xavier_uniform_(self.tovalue)
        nn.init.xavier_uniform_(self.unifyheads)
        nn.init.uniform_(self.unifyheads_bias, a=-l, b=l)
        nn.init.zeros_(self.tovalue_bias)
        nn.init.zeros_(self.tokey_bias)
        nn.init.zeros_(self.toquery_bias)

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    b, c, l, h = *x.size(), self.heads

    # Compute inner products between keys and queries
    key = torch.einsum('bki,hek->bhei', x, self.tokey) + self.tokey_bias
    query = torch.einsum('heq,bqj->bhej', self.toquery, x) + self.toquery_bias
    weights = torch.einsum('bhei, bhej -> bhij', key, query) / math.sqrt(self.inner_dim)

    #weights = torch.einsum('bki,hek,heq,bqj->bhij', x, self.tokey, self.toquery, x) / math.sqrt(c)

    weights = self.softmax(weights)

    #if self.value_identity:
    #    x = einops.repeat(x, 'b v i -> b h v i', h = h)
    #else:
        # Transform input to value
    x = torch.einsum('bvi,htv->bhti', x, self.tovalue) + self.tovalue_bias

    # Compute the weighted sum of the value
    x = torch.einsum('bhij,bhti->bhtj', weights, x)

    # Unify the individual heads
    x = torch.einsum('bhtl,hct->bcl', x, self.unifyheads) + self.unifyheads_bias

    return x
