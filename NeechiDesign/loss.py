import torch
from config import *
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__() 
        self.temperature = temperature  
        self.device = device     
    def forward(self, x, y):
        loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        for xID in range(x.shape[0]):
            neg_dot = torch.zeros(y.shape[0]).to(self.device)
            for yID in range(y.shape[0]):
                neg_dot[yID] = x[xID].dot(y[yID])
                exp = torch.exp(neg_dot/self.temperature)
                # Regard the corresponding sentence-eegSequence as a positive pair
                loss += - torch.log(exp[xID]/torch.sum(exp))
        return loss

# FIXME: 'torch.linalg.eigh' can not be used to calculate grad --- backward found 'nan'
class cca_loss(nn.Module):
    def __init__(self, outdim_size, use_all_singular_values, device):
        super().__init__() 
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def forward(self, x, y):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = x.t(), y.t()

        o1 =  H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)


        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        try:
            [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO='U')
            [D2, V2] = torch.linalg.eigh(SigmaHat22, UPLO='U')
        except torch._C._LinAlgError:
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        # Original
        # [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        # [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        D1 = D1.unsqueeze(1)
        posInd1 = torch.gt(D1[:, 0], eps).nonzero()[:, 0]
        D1 = D1[posInd1, 0]
        V1 = V1[:, posInd1]
        D1 = torch.squeeze(D1)  # Remove extra dimensions from D1
        # Reshape D1 to 1-dimensional tensor
        D2 = D2.unsqueeze(1)
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        posInd2 = torch.gt(D2[:, 0], eps).nonzero()[:, 0]
        D2 = D2[posInd2, 0]
        V2 = V2[:, posInd2]
        D2 = torch.squeeze(D2)
        
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)


        if self.use_all_singular_values:
    
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
  
        else:
   
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) 
            U, V = torch.linalg.eigh(trace_TT, UPLO='U')
            U = torch.where(U>eps, U, (torch.ones(U.shape).float()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr            # tensor(-0.4951, device='cuda:5', grad_fn=<NegBackward0>)
# cca = cca_loss(3, False, device='cuda:5')

def cal_loss(cca_weight=1, wd_weight=1, text_embed=None, eeg_embed=None):
    # loss = cca_weight * cca(text_embed, eeg_embed)
    loss = wd_weight * torch.tensor(wasserstein_distance(text_embed.cpu().detach().numpy().flatten(), eeg_embed.cpu().detach().numpy().flatten()), requires_grad=True)
    return loss