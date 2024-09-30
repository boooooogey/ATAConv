'''
    Implementation of the Nystrom layer proposed in https://arxiv.org/abs/1911.13036.
'''
import torch
from torch.nn import Module
from torch.nn import Linear
from torch import no_grad
from torch.nn import Parameter
from einops import einsum
from kmeans_pytorch import kmeans


def linear_kernel(input_x, input_y):
    '''
        Calculate linear kernel between two sample group.
    '''
    return einsum(input_x, input_y, '... i j, ... h j -> ... h i')

def inverse_sqrt(mat):
    eig = torch.linalg.eig(mat)
    inv_sqrt = einsum(eig.eigenvectors, 1/torch.sqrt(eig.eigenvalues), eig.eigenvectors, "i k, k, j k -> i j")
    return torch.real(inv_sqrt)

class Nystrom(Module):
    '''
        Implementation of Nystrom layer. It uses a subset of the samples in a batch to calculate
        Nystrom features for the rest of the samples in the batch. The inverse of the gram matrix
        among the sampled are learned adaptively instead of being computed. 
    '''
    def __init__(self, n_sample, kernel = None):
        super().__init__()
        self.n_sample = n_sample
        if kernel is None:
            self.kernel = linear_kernel
        self.adaptive_nystrom = Linear(in_features = n_sample,
                                       out_features = n_sample,
                                       bias = False)

    def forward(self, input_x, **kwargs):
        '''
            Use a random permutation to select n_sample samples. Then calculate the gram matrix
            between selected and the rest and apply the adaptive Nystrom weigthts.
        '''
        if input_x.shape[0] < self.n_sample:
            print("Warning the sample size is larger than batch size.")
        else:
            _, cluster_centers = kmeans(X = input_x,
                                        num_clusters = self.n_sample,
                                        distance='euclidean',
                                        device = input_x.get_device(),
                                        tqdm_flag = False
            )
            cluster_centers = cluster_centers.to(input_x.get_device())
            gram = self.kernel(cluster_centers, input_x, **kwargs)
            return self.adaptive_nystrom(gram), cluster_centers

class Nystrom_Feature(Module):
    '''
        Implementation of Nystrom layer. It uses a subset of the samples in a batch to calculate
        Nystrom features for the rest of the samples in the batch. The inverse of the gram matrix
        among the sampled are learned adaptively instead of being computed. 
    '''
    def __init__(self, feature_dim, kernel = None):
        super().__init__()
        self.feature_dim = feature_dim 
        if kernel is None:
            self.kernel = linear_kernel
        self.adaptive_nystrom = Linear(in_features = feature_dim,
                                       out_features = feature_dim,
                                       bias = False)

    def forward(self, input_x, **kwargs):
        '''
            Use a random permutation to select feature_dim samples. Then calculate the gram matrix
            between selected and the rest and apply the adaptive Nystrom weigthts.
        '''
        b, f, l = input_x.shape
        dev = input_x.get_device()
        if f < self.feature_dim:
            print("Warning the feature dimension is larger than possible samples (batch x feature).")
        else:
            with no_grad():
                _, cluster_centers = kmeans(X = input_x[0],
                                            num_clusters = self.feature_dim,
                                            distance='euclidean',
                                            device = dev if dev >= 0 else "cpu",
                                            tqdm_flag = False
                )
                if dev > -1:
                    cluster_centers = cluster_centers.to(input_x.get_device())
            gram = self.kernel(cluster_centers, input_x, **kwargs) / l
            return self.adaptive_nystrom(gram), cluster_centers
    
    def gram_approximation(self, input_x, **kwargs):
        with no_grad():
            dev = input_x.get_device()
            _, cluster_centers = kmeans(X = input_x[0],
                                        num_clusters = self.feature_dim,
                                        distance='euclidean',
                                        device = dev if dev >= 0 else "cpu",
                                        tqdm_flag = False
            )
            if dev > -1:
                cluster_centers = cluster_centers.to(input_x.get_device())
            gram = self.kernel(cluster_centers, input_x, **kwargs)
            return einsum(gram, self.adaptive_nystrom.weight, self.adaptive_nystrom.weight, gram,
                          '... i j, j k, k j, ... h j -> ... h i')

class Nystrom_Sampling(Module):
    '''
        Implementation of Nystrom where the archetype of samples are learned. Then learned
        archetypes are used to calculate the Nystrom feature map.
    '''
    def __init__(self, length, feature_dim, kernel = None):
        super().__init__()
        self.feature_dim = feature_dim 
        self.length = length 
        if kernel is None:
            self.kernel = linear_kernel
        self.archetype = Parameter(torch.empty(feature_dim, length))
        torch.nn.init.kaiming_uniform_(self.archetype, nonlinearity='linear')

    def forward(self, input_x, **kwargs):
        '''
            forward.
        '''
        k21 = self.kernel(self.archetype, input_x, **kwargs)
        k11 = self.kernel(self.archetype, self.archetype, **kwargs)
        k11_inv_sqrt = inverse_sqrt(k11)
        return einsum(k21, k11_inv_sqrt, "... l, l f -> ... f")
    
    def gram_approximation(self, input_x, **kwargs):
        '''
            Approximate kernel matrix using learned archetypes
        '''
        with no_grad():
            k21 = self.kernel(self.archetype, input_x, **kwargs)
            k11 = self.kernel(self.archetype, self.archetype, **kwargs)
            k11_inv_sqrt = inverse_sqrt(k11)
            return einsum(k21, k11_inv_sqrt, k11_inv_sqrt, k21,
                          "... x l, l k, k m, ... y m -> ... x y")



if __name__ == "__main__":
    import torch
    layer = Nystrom_Sampling(300, 4)
    x = torch.rand(3, 100, 300)
    layer(x).shape