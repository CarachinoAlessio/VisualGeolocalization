import torch
import torch.nn as nn
from torch.nn import Parameter


def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class ArcFaceLoss(nn.Module):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, in_features: int, out_features: int, s: float = 64., m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight) 

        with torch.no_grad():
            theta_m = torch.acos(cosine.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, label.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cosine

        logits = self.s * (cosine + d_theta)
        return logits
    
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'