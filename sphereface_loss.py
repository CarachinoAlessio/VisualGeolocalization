import torch
import torch.nn as nn
from torch.nn import Parameter
import math

def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)                      
    return ip / torch.ger(w1, w2).clamp(min=eps)


class SphereFaceLoss(nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
    It also used characteristic gradient detachment tricks proposed in
    <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 1.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cos_theta = cosine_sim(inputs, self.weight)

        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            m_theta.scatter_(1, label.view(-1, 1), self.m, reduce = 'multiply')
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - cos_theta

        logits = self.s * (cos_theta + d_theta)
        return logits

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'