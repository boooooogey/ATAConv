import einops
import torch
from torch import nn

class FastAttention(nn.Module):
    def __init__(
        self,
        num_channel,
        num_heads = 8,
        embedding_size = None,
        nb_features = None,
        kernel_fn = nn.ReLU(),
        feature_redraw_interval = 1000
    ):
        super().__init__()

        self.feature_redraw_interval = feature_redraw_interval
        self.timer = 0

        if embedding_size is None:
            embedding_size = num_channel

        self.fast_attention = FastAttention(embedding_size, nb_features, causal = False, generalized_attention = False, kernel_fn = kernel_fn, no_projection = False)

        self.num_heads = num_heads
        self.embedding_size = embedding_size
        

        self.tokey = nn.Parameter(torch.empty(num_heads, embedding_size, num_channel))
        self.toquery = nn.Parameter(torch.empty(num_heads, embedding_size, num_channel))
        self.tovalue = nn.Parameter(torch.empty(num_heads, num_channel, num_channel))
        self.toout = nn.Parameter(torch.empty(num_heads, num_channel, num_channel))

        #Initialize the weights
        with torch.no_grad():
            nn.init.xavier_uniform_(self.tokey)
            nn.init.xavier_uniform_(self.toquery)
            nn.init.xavier_uniform_(self.tovalue)
            nn.init.xavier_uniform_(self.toout)

    def forward(self, x, pos_emb = None, **kwargs):

        self.redraw_projection()

        b, c, l, h, e = *x.shape, self.num_heads, self.embedding_size
        
        key = einops.einsum(x, self.tokey, 'b c l, h e c -> b h l e')
        query = einops.einsum(x, self.toquery, 'b c l, h e c -> b h l e')
        value = einops.einsum(x, self.tovalue, 'b c l, h e c -> b h l e')

        #if exists(pos_emb) and not cross_attend:
        #    q, k = apply_rotary_pos_emb(q, k, pos_emb)

        out = self.fast_attention(query, key, value)

        out = einops.einsum(out, self.toout, 'b h l e, h c e -> b c l')
        return out

    def redraw_projection(self):
        if self.feature_redraw_interval is not None:
            if self.timer >= self.feature_redraw_interval:
                device = get_module_device(self)
                self.fast_attention.redraw_projection_matrix(device)
                self.timer = 0
            else:
                self.timer += 1 

