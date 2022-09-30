import torch

class AttentionPooling1D(torch.nn.Module):
  def __init__(self,
               kernel_size,
               feature_size,
               init_constant = 2,
               separate_channel = False):
    '''
    If separate_channel = True then it is similar to max pooling
    If separate_channel = False then it is similar to average pooling
    '''
    super(AttentionPooling1D, self).__init__()

    self.kernel_size = kernel_size
    self.feature_size = feature_size

    self.logit_linear = torch.nn.Linear(in_features=feature_size, 
                                        out_features=feature_size if separate_channel else 1,
                                        bias=False)

    with torch.no_grad():
      torch.nn.init.eye_(self.logit_linear.weight)#Avg
      self.logit_linear.weight.mul_(init_constant)
    self.softmax = torch.nn.Softmax(dim=-1)

  def forward(self, x):
    b, c, l = x.shape
    kernel_size = self.kernel_size
    if l % kernel_size != 0:
      pad_size = kernel_size - l % kernel_size 
      pad_left = kernel_size // 2
      pad_right = pad_size - pad_left

      x = torch.nn.functional.pad(x, (pad_left, pad_right), mode="constant", value=1/self.feature_size)
      l = x.shape[2]

    x = x.view(b, c, l//kernel_size, kernel_size)

    mask = torch.einsum("bcld, ec -> beld", x, self.logit_linear.weight)
    mask = self.softmax(mask)

    return torch.sum(x * mask, axis=-1), mask
