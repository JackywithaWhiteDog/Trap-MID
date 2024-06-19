import torch
from torch.nn import functional as F

def ls_scheduler(epoch, n_epochs):
    if epoch < 50:
        return 0.0
    if epoch < 75:
        factor = 1 / (n_epochs - 50)
        return factor * (epoch + 1 - 50)
    return 1

def mnist_ls_scheduler(epoch, n_epochs):
    if epoch < 5:
        return 0.0
    if epoch < 7:
        factor = 1 / (n_epochs - 5)
        return factor * (epoch + 1 - 5)
    return 1

class NegLSCrossEntropyLoss(torch.nn.Module):
    def __init__(self, init_ls=0.0, scheduler=ls_scheduler):
        super().__init__()
        self.init_ls = init_ls
        self.label_smoothing = init_ls
        self.scheduler = scheduler

    def step(self, epoch, n_epochs):
        if self.scheduler is None:
            return
        self.label_smoothing = self.init_ls * self.scheduler(epoch, n_epochs)

    def forward(self, logits, labels):
        confidence = 1.0 - self.label_smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        return torch.sum(loss) / num_batch
