from torch.autograd import Variable
import torch
import torch.nn as nn

class GumbelSoftmax(nn.Module):
    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor + eps) - torch.log(
            1 - uniform_samples_tensor + eps)
        return gumble_samples_tensor

    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits

    def forward(self, logits):
        if not self.training:
            out_hard = (logits >= 0).float()
            return out_hard, None
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard, prob_soft