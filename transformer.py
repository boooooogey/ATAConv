import torch
from torch import nn
import torch.nn.functional as F
from IPython import embed
import math

class SelfAttention(nn.Module):
  def __init__(self, k, heads=8):
    super().__init__()
    self.k, self.heads = k, heads

    self.tokeys = nn.Linear(k, k * heads, bias=False)
    self.toqueries = nn.Linear(k, k * heads, bias=False)
    self.tovalues = nn.Linear(k, k * heads, bias=False)
    self.unifyheads = nn.Linear(heads * k, k)

  def forward(self, x):
    b, k, t = x.size()
    h = self.heads

    keys = torch.einsum('bkt,jk->bjt', x, self.tokeys.weight).view(b, h, k, t)
    queries = torch.einsum('bkt,jk->bjt', x, self.toqueries.weight).view(b, h, k, t)
    values = torch.einsum('bkt,jk->bjt', x, self.tovalues.weight).view(b, h, k, t)

    dot = torch.einsum('bhkt,bhki->bhti', queries, keys) / math.sqrt(k)
    dot = F.softmax(dot, dim=-1)

    out = torch.einsum('bhti,bhki->bhkt', dot, values)
    out = torch.einsum('bhet,khe->bkt', out, self.unifyheads.weight.view(k,h,k))

    return out + self.unifyheads.bias.view(1,-1,1)

#  def forward(self, x):
#    b, t, k = x.size()
#    h = self.heads
#
#    keys = self.tokeys(x).view(b, t, h, k)
#    queries = self.toqueries(x).view(b, t, h, k)
#    values = self.tovalues(x).view(b, t, h, k)
#
#    #dot = torch.einsum('bthe,bihe->bhti', queries, keys) / math.sqrt(e)
#    #dot = F.softmax(dot, dim=-1)
#
#    keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
#    queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
#    values = values.transpose(1, 2).contiguous().view(b * h, t, k)
#
#    queries = queries / (k ** (1/4))
#    keys    = keys / (k ** (1/4))
#
#    dot = torch.bmm(queries, keys.transpose(1, 2))
#
#    dot = F.softmax(dot, dim=2)
#
#    #out = torch.einsum('bhtd,bdhe->bthe', dot, values)
#
#    #out = torch.einsum('bthe,khe->btk', out, self.unifyheads.weight.view(e,h,e))
#
#    out = torch.bmm(dot, values).view(b, h, t, k)
#
#    out = out.transpose(1, 2).contiguous().view(b, t, h * k)
#
#    return self.unifyheads(out)

class TransformerBlock(nn.Module):
  def __init__(self, k, heads):
    super().__init__()

    self.attention = SelfAttention(k, heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, 4 * k),
      nn.ReLU(),
      nn.Linear(4 * k, k))

  def forward(self, x):
    attended = self.attention(x)
    attended = attended.transpose(1,2)
    x = self.norm1(attended + x.transpose(1,2))

    fedforward = self.ff(x)
    return self.norm2(fedforward + x).transpose(1,2)

if __name__ == "__main__":
  x = torch.rand(64, 3, 100)
  atl = SelfAttention(3)
  attention = atl.forward(x)
  print(x.shape)
  print(attention.shape)
