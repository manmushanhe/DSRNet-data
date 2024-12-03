from collections import defaultdict

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper

# 固定顺序解决器
class FixedOrderSolver(AbsLossWrapper):
    def __init__(self, criterion: AbsEnhLoss, weight=1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, ref, inf, others={}):
        """An naive fixed-order solver

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk   参考语音(干净语音)
            inf (List[torch.Tensor]): [(batch, ...), ...]           推理语音
   
        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation  使用最好的置换最小的loss
            stats: dict, for collecting training status
            others: reserved
            criterion为对应的tf_domain中的FrequencyDomainMSE
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = 1

        loss = 0.0
        stats = defaultdict(list)
        #单说话人
        #self.criterion(ref, inf)返回loss
        '''
        for r, i in zip(ref, inf):
            loss += torch.mean(self.criterion(r, i)) / num_spk
            for k, v in getattr(self.criterion, "stats", {}).items():
                stats[k].append(v)

        for k, v in stats.items():
            stats[k] = torch.stack(v, dim=1).mean()
        stats[self.criterion.name] = loss.detach()
        '''
        # self.criterion(ref, inf).size() = (Batch,)
        # torch.mean(self.criterion(ref, inf)).size() = (1,)
        loss += torch.mean(self.criterion(ref, inf)) 
        
        for k, v in getattr(self.criterion, "stats", {}).items():
            stats[k].append(v)
        
        for k, v in stats.items():
            stats[k] = torch.stack(v, dim=1).mean()
        stats[self.criterion.name] = loss.detach()
        # loss.mean().size() = (1,)
        return loss.mean(), dict(stats), {}
