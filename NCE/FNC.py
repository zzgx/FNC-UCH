import pdb
import torch
from torch import nn
import math
import torch.nn.functional as F

class FNC(nn.Module):
    def __init__(self, bit, n_data, TOP_FNPS, threshold):
        super(FNC, self).__init__()
        self.K = 2500
        self.TOP_FNPS = TOP_FNPS
        self.threshold = threshold
        self.T = 0.9 * math.sqrt(bit)
        self.momentum = 0.4
        stdv = 1. / math.sqrt(bit / 3)
        rnd = torch.randn(n_data, bit).mul_(2 * stdv).add_(-stdv)
        self.memory = F.normalize(rnd.sign(), dim=1).cuda()

        print(f'FNC')
        print(f'K={self.K}')
        print(f'TOP_FNPS={self.TOP_FNPS}')
        print(f'threshold={self.threshold}')

    def forward(self, i_A, i_B, t_A, t_B, batch_idx, warmup_count=0):
        K = int(self.K)
        T = self.T
        if warmup_count > 0:
            momentum = 0
        else:
            momentum = self.momentum
        memory = self.memory
        batchSize = batch_idx.size(0)
        bit = memory.size(1)
        total_memory = memory.size(0)
        TOP_FNPS = self.TOP_FNPS
        threshold = self.threshold

        # batch(warmup)
        if momentum == 0:
            weight = (i_A + t_A) / 2.
            inx = torch.stack([torch.arange(batchSize)] * batchSize)
            inx = torch.cat(
                [torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])],
                dim=1).to(weight.device).view([-1])
            weight = weight[inx].view([batchSize, batchSize, -1])
            weight = weight.sign_()
            sim_I = torch.bmm(weight, i_A.view(batchSize, bit, 1))
            sim_T = torch.bmm(weight, t_A.view(batchSize, bit, 1))
            sim_I = torch.div(sim_I, T)
            sim_I = sim_I.contiguous()
            sim_T = torch.div(sim_T, T)
            sim_T = sim_T.contiguous()
            sim_I = sim_I.softmax(1)
            sim_T = sim_T.softmax(1)
            I_loss = -sim_I[:, 0].log().mean()
            T_loss = -sim_T[:, 0].log().mean()
            avg_loss = I_loss + T_loss

        # memory
        else:

            fusion1 = (i_A + t_A) / 2.
            fusion2 = (i_A + t_B) / 2.
            fusion3 = (i_B + t_A) / 2.
            fusion4 = (i_B + t_B) / 2.


            fusion1_norm = F.normalize(fusion1, p=2, dim=1)
            fusion2_norm = F.normalize(fusion2, p=2, dim=1)
            fusion3_norm = F.normalize(fusion3, p=2, dim=1)
            fusion4_norm = F.normalize(fusion4, p=2, dim=1)


            sim1 = torch.mm(memory, fusion1_norm.t())
            sim2 = torch.mm(memory, fusion2_norm.t())
            sim3 = torch.mm(memory, fusion3_norm.t())
            sim4 = torch.mm(memory, fusion4_norm.t())


            final_sim = (sim1 + sim2 + sim3 + sim4) / 4.



            batch_loss = 0.
            for i in range(batchSize):

                sim_column = final_sim[:, i]


                sorted_idx = torch.argsort(sim_column, descending=True)  # descending=True 表示从大到小


                pos_idx = batch_idx[i:i + 1]


                sorted_idx_topk = sorted_idx[:TOP_FNPS]
                sorted_sim_topk = sim_column[sorted_idx_topk]
                mask = sorted_sim_topk > threshold
                fnps_idx = sorted_idx_topk[mask][:TOP_FNPS]
                filtered_fnps_idx = fnps_idx[fnps_idx != pos_idx.item()]


                high = int(total_memory * 0.90)
                Negative_candidates = sorted_idx[0:high]
                exclude_idx = torch.cat([filtered_fnps_idx, pos_idx])
                mask_2 = ~torch.isin(Negative_candidates, exclude_idx)
                Negative_candidates = Negative_candidates[mask_2]
                Negative_selected = torch.randperm(len(Negative_candidates))[:K]
                Negative_idx = Negative_candidates[Negative_selected]


                anchor = fusion1[i]


                pos_samples = self.memory[pos_idx]
                fnp_samples = self.memory[filtered_fnps_idx]


                neg_samples = self.memory[Negative_idx]


                pos_samples = pos_samples.sign_()
                fnp_samples = fnp_samples.sign_()
                neg_samples = neg_samples.sign_()


                pos_sim = torch.matmul(pos_samples, anchor) / T
                fnp_sim = torch.matmul(fnp_samples, anchor) / T
                neg_sim = torch.matmul(neg_samples, anchor) / T


                pos_exp = torch.exp(pos_sim)
                fnp_exp = torch.exp(fnp_sim)
                neg_exp = torch.exp(neg_sim)



                beta_weights = fnp_sim
                alpha_weights = 1 - beta_weights

                numerator = pos_exp.sum() + (beta_weights * fnp_exp).sum()
                denominator = neg_exp.sum() + pos_exp.sum()


                batch_loss += (- torch.log(numerator / denominator) / (1 + len(fnp_samples)))


            avg_loss = batch_loss / batchSize


        with torch.no_grad():
            new_feature = (i_A + t_A) / 2.
            new_feature = F.normalize(new_feature, p=2, dim=1)

            old_feature = torch.index_select(memory, 0, batch_idx.view(-1))

            old_feature.mul_(momentum)

            old_feature.add_(torch.mul(new_feature, 1 - momentum))

            old_feature = F.normalize(old_feature, p=2, dim=1)

            memory.index_copy_(0, batch_idx, old_feature)

        return avg_loss