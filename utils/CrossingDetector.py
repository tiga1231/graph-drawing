from torch import nn


## neural crossing detector
class CrossingDetector(nn.Module):
    def __init__(self):
        super().__init__()
#         self.layer_dims = [8,256,1024,128,1]
        self.layer_dims = [8,128,512,64,1]
        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < len(self.layer_dims)-2:
                self.layers.append(nn.LayerNorm(out_dim))
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.main(x)