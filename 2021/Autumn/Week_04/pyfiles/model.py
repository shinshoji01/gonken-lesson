import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)
    
class Expand(nn.Module):
    def __init__(self, H, W):
        super(Expand, self).__init__()
        self.H = H
        self.W = W

    def forward(self, x):
        return x.reshape(x.size(0), self.H, self.W)
    
class VAE(nn.Module):
    
    def __init__(self, z_dim, nch_input, n_layers=4, mul=4, nch=32, device="cpu", beta=1, image_shape=(32,32)):
        super(VAE, self).__init__()
        self.device = device
        self.beta = beta
        self.image_shape = image_shape
        
        # Encoding
        layers = []
        in_features = image_shape[0]*image_shape[1]*1
        out_features = nch*(mul**(n_layers-2))
        layers.append(Flatten())
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        
        in_features = out_features
        for _ in range(n_layers-2):
            out_features = in_features//mul
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
            
        self.encoder = nn.Sequential(*layers)
        self.encmean = nn.Linear(out_features, z_dim)
        self.encvar = nn.Linear(out_features, z_dim)
        
        # Decoding
        
        layers = []
        in_features = z_dim
        out_features = nch
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        
        in_features = out_features
        for _ in range(n_layers-2):
            out_features = in_features*mul
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
            
        in_features = nch*(mul**(n_layers-2))
        out_features = image_shape[1]*image_shape[0]*1
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Tanh())
        layers.append(Expand(image_shape[0], image_shape[1]))
        
        self.decoder = nn.Sequential(*layers)
        
    def _encoder(self, x):
        for layer in self.encoder:
            x = layer(x)
        mean = self.encmean(x)
        var = F.softplus(self.encvar(x))
        return mean, var
    
    def _decoder(self, z):
        for layer in self.decoder:
            z = layer(z)
        z = torch.tanh(z)
        z = torch.clamp(z, min=-1+1e-8, max=1-1e-8)
        return z
    
    def _samplez(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon
    
    def forward(self, x, get_loss=False):
        mean, var = self._encoder(x)
        z = self._samplez(mean, var)
        y = self._decoder(z)
        if get_loss:
            # KL divergence
            KL = -0.5 * torch.sum(1 + torch.log(var+1e-8) - mean**2 - var)  / x.shape[0]
            # reconstruction error
            recon = F.mse_loss(y.view(-1), x.view(-1), size_average=False) / x.shape[0]
            # combine them
            lower_bound = [self.beta*KL, recon]
            return y, z, sum(lower_bound)
        else:
            return y, z