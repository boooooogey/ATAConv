import torch
from torch.nn import Parameter

class AttentionPooling1D(torch.nn.Module):
  def __init__(self,
               kernel_size,
               feature_size,
               init_constant = 2,
               keep_shape = False,
               mode = "diagonal" # "full", "diagonal", "shared"
               ):
    '''
    if mode == diagonal: it is not an attention layer.
    if mode == full: it can achieve maximum
    '''
    super(AttentionPooling1D, self).__init__()

    self.kernel_size = kernel_size
    self.feature_size = feature_size
    self.mode = mode
    self.keep_shape = keep_shape

    if mode != "diagonal":
       self.out_feature = feature_size if mode != "shared" else 1 
       self.weight = Parameter(torch.empty(self.out_feature, feature_size, requires_grad = True)) 
       with torch.no_grad():
         if mode == "full":
           torch.nn.init.eye_(self.weight)
           self.weight.mul_(init_constant)
         if mode == "shared":
           nn.init.xavier_uniform_(self.weight)
    else:
       self.out_feature = feature_size
       self.weight = Parameter(torch.empty(feature_size, requires_grad = True))
       with torch.no_grad():
         torch.nn.init.ones_(self.weight)
         self.weight.mul_(init_constant)

    self.softmax = torch.nn.Softmax(dim=-1)

  def forward(self, x):
    b, c, l = x.shape
    original_length = l
    kernel_size = self.kernel_size
    if l % kernel_size != 0:
      #sub_length = (l//kernel_size) * kernel_size
      #x = x[:, :, :sub_length]
      #l = sub_length 
      reminder = kernel_size - l % kernel_size
      x = torch.nn.functional.pad(x, (0,reminder,0,0,0,0))
      _, _, l = x.shape

    x = x.view(b, c, l//kernel_size, kernel_size)

    if self.mode != "diagonal":
        mask = torch.einsum("bcld, ec -> beld", x, self.weight)
        mask = self.softmax(mask)
    else:
        mask = torch.einsum("bcld, c -> bcld", x, self.weight)
        mask = self.softmax(mask)

    if self.keep_shape:
        out = torch.sum(x * mask, axis=-1).repeat_interleave(kernel_size, dim=-1)[:,:,:original_length]
    else:
        out = torch.sum(x * mask, axis=-1)

    return out, mask
