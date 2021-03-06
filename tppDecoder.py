import torch
from torch.nn import ModuleList, Linear, Parameter
from torch_geometric.nn.inits import zeros
import numpy as np


class DyRepDecoder(torch.nn.Module):
    def __init__(self, embedding_dim, num_surv_samples, train_td_hr_max, train_td_max, time_type='hr'):
        super(DyRepDecoder, self).__init__()
        self.embed_dim = embedding_dim
        self.num_surv_samples = num_surv_samples
        self.train_td_hr_max = train_td_hr_max
        self.train_td_max = train_td_max
        self.time_type = time_type
        self.omega = ModuleList([Linear(in_features=2*self.embed_dim, out_features=1),
                                Linear(in_features=2 * self.embed_dim, out_features=1)])

        self.psi = Parameter(0.5*torch.ones(2))
        self.alpha = Parameter(torch.rand(2))
        self.w_t = Parameter(torch.rand(2))
        self.reset_parameters()


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def forward(self, all_embeddings, assoc, src, pos_dst, neg_dst_surv,
                neg_src_surv=None, neg_dst=None, last_update=None, cur_time=None, et=None, tp=False):

        z_src, z_dst = all_embeddings[assoc[src]], all_embeddings[assoc[pos_dst]]
        z_neg_dst_surv = all_embeddings[assoc[neg_dst_surv]]

        surv_v = torch.zeros(z_neg_dst_surv.size(0))
        if last_update is None:
            lambda_uv = self.compute_intensity_lambda(z_src, z_dst)

            surv_u = self.compute_intensity_lambda(
                z_src.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
                z_neg_dst_surv)
            if neg_src_surv is not None:
                z_neg_src_surv = all_embeddings[assoc[neg_src_surv]]
                surv_v = self.compute_intensity_lambda(
                    z_neg_src_surv,
                    z_dst.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim))
        else:
            last_time_pos = torch.cat((last_update[assoc[src]].view(-1,1),
                                       last_update[assoc[pos_dst]].view(-1,1)), dim=1).max(-1)[0]
            last_time_neg_dst_surv = torch.cat((last_update[assoc[src]].unsqueeze(1).repeat(1, self.num_surv_samples).view(-1,1),
                                       last_update[assoc[neg_dst_surv]].view(-1,1)), dim=1).max(-1)[0]
            td_pos = cur_time - last_time_pos
            td_neg_dst = cur_time.unsqueeze(1).repeat(1, self.num_surv_samples).view(-1) - last_time_neg_dst_surv

            lambda_uv = self.hawkes_intensity(z_src, z_dst, et, td_pos)

            surv_u_0 = self.hawkes_intensity(
                z_src.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
                z_neg_dst_surv, torch.zeros(len(z_neg_dst_surv)), td_neg_dst)
            # surv_u_0 = 0
            surv_u_1 = self.hawkes_intensity(
                z_src.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
                z_neg_dst_surv, torch.ones(len(z_neg_dst_surv)), td_neg_dst)
            surv_u = surv_u_0 + surv_u_1

            if neg_src_surv is not None:
                z_neg_src_surv = all_embeddings[assoc[neg_src_surv]]
                last_time_neg_src_surv = \
                    torch.cat((last_update[assoc[neg_src_surv]].view(-1,1),
                               last_update[assoc[pos_dst]].unsqueeze(1).repeat(1, self.num_surv_samples).view(-1,1)),
                              dim=1).max(-1)[0]
                td_neg_src = cur_time.unsqueeze(1).repeat(1, self.num_surv_samples).view(-1) - last_time_neg_src_surv

                surv_v_0 = self.hawkes_intensity(
                    z_neg_src_surv,
                    z_dst.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
                    torch.zeros(len(z_neg_src_surv)), td_neg_src)

                surv_v_1 = self.hawkes_intensity(
                    z_neg_src_surv,
                    z_dst.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
                    torch.ones(len(z_neg_src_surv)), td_neg_src)
                surv_v = surv_v_0 + surv_v_1

        loss_lambda = -torch.sum(torch.log(lambda_uv + 1e-7))
        # loss_surv = ((torch.sum(surv_u)+torch.sum(surv_v)) / self.num_surv_samples)
        loss_surv_u = torch.sum(surv_u) / self.num_surv_samples
        loss_surv_v = torch.sum(surv_v) / self.num_surv_samples

        cond_density = []
        if (not self.training) or (neg_dst is not None):
            with torch.no_grad():
                z_neg_dst = all_embeddings[assoc[neg_dst]]
                if last_update is None:
                    lambda_uv_neg = self.compute_intensity_lambda(z_src, z_neg_dst)
                else:
                    last_time_neg = torch.cat((last_update[assoc[src]].view(-1,1), last_update[assoc[neg_dst]].view(-1,1)), dim=1).max(-1)[0]
                    td_neg = cur_time - last_time_neg
                    lambda_uv_neg = self.hawkes_intensity(z_src, z_neg_dst, et, td_neg)
                s = surv_u.view(-1, self.num_surv_samples).mean(dim=-1) + surv_v.view(-1, self.num_surv_samples).mean(dim=-1)
                surv = torch.exp(-s)
                assert len(z_src) == len(surv)
                cond_pos = lambda_uv * surv
                cond_neg = lambda_uv_neg * surv
                cond_density = [cond_pos, cond_neg]
        # return loss_lambda/len(z_src), loss_surv/len(z_src)
        return loss_lambda/len(z_src), loss_surv_u/len(z_src), loss_surv_v/len(z_src), cond_density

    def hawkes_intensity(self, z_u, z_v, et_uv, td, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        et = (et_uv > 0).long()
        if self.time_type == 'hr':
            td_norm = td / self.train_td_hr_max
        else:
            td_norm = td / self.train_td_max
            # td_norm = (td - train_td_mean) / train_td_std
        if symmetric:
            z_uv = torch.cat((z_u, z_v), dim=1)
            z_vu = torch.cat((z_v, z_u), dim=1)
            g_uv = z_uv.new_zeros(len(z_uv))
            g_vu = z_vu.new_zeros(len(z_vu))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g_uv[idx] = self.omega[k](z_uv).flatten()[idx]
                    g_vu[idx] = self.omega[k](z_vu).flatten()[idx]
                    # g_uv = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
            # g_vu = self.omega(torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            z_cat = torch.cat((z_u, z_v), dim=1)
            g = z_cat.new_zeros(len(z_cat))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g[idx] = self.omega[k](z_cat).flatten()[idx]

            # g = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
        psi = self.psi[et]
        alpha = self.alpha[et]
        w_t = self.w_t[et]
        g += alpha*torch.exp(-w_t*td_norm)
        # g_psi = g / (psi + 1e-7)
        g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) #+ alpha*torch.exp(-w_t*td_norm)
        return Lambda



class DecoderTP(torch.nn.Module):
    def __init__(self, num_nodes, start_t, embedding_dim, num_surv_samples, train_td_hr_max, train_td_max,
                 train_td_hr_mean=None, train_td_hr_std=None, num_rel=2, time_type='hr', unit_sampling=True, decay_out=False,
                 device='cpu'):
        super(DecoderTP, self).__init__()
        self.unit_sampling = unit_sampling
        self.embed_dim = embedding_dim
        self.num_surv_samples = num_surv_samples
        self.train_td_hr_max = train_td_hr_max

        self.train_td_max = train_td_max
        self.train_td_hr_mean = train_td_hr_mean
        self.train_td_hr_std = train_td_hr_std
        self.time_type = time_type
        self.decay_out = decay_out
        self.device=device
        self.num_rel = num_rel
        self.omega = ModuleList([Linear(in_features=2*self.embed_dim, out_features=1) for _ in range(num_rel)])

        z_last_memory = torch.empty(num_nodes, embedding_dim, device=device)
        last_last_time = torch.zeros(num_nodes, dtype=start_t.dtype, device=device)
        self.register_buffer('z_last_memory', z_last_memory)
        self.register_buffer('last_last_time', last_last_time)

        self.psi = Parameter(0.5*torch.ones(num_rel))
        # self.alpha = Parameter(torch.rand(num_rel))
        self.alpha = Parameter(-0.1*torch.ones(num_rel))
        # self.w_t = Parameter(torch.ones(num_rel))
        self.w_t = Parameter(torch.rand(num_rel))
        self.reset_parameters()


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.z_last_memory)
        zeros(self.last_last_time)


    def forward(self, all_embeddings, assoc, src, pos_dst, last_update, cur_time,
                u_non_embeddings, v_non_embeddings, neg_dst=None, et=None, last_time_pos=None, accu_non_events=None):

        z_src, z_dst = all_embeddings[assoc[src]], all_embeddings[assoc[pos_dst]]

        if last_time_pos is None:
            last_time_pos = torch.cat((last_update[assoc[src]].view(-1,1),
                                       last_update[assoc[pos_dst]].view(-1,1)), dim=1).max(-1)[0]
            td_pos = cur_time - last_time_pos
        else:
            td_pos = cur_time - last_time_pos
            t_uv = torch.cat((last_update[assoc[src]].view(-1,1), last_update[assoc[pos_dst]].view(-1,1),
                              last_time_pos.view(-1,1)), dim=1).max(-1)[0]
            last_uv = torch.cat((last_update[assoc[src]].view(-1,1),
                                 last_update[assoc[pos_dst]].view(-1,1)), dim=1).max(-1)[0]
            td_uv = cur_time - t_uv

        if self.unit_sampling:
            td_surv_step = td_pos / (self.num_surv_samples-1)
            td_surv_base = torch.linspace(0, self.num_surv_samples-1, self.num_surv_samples).unsqueeze(1) \
                .repeat(1, len(src)).to(self.device)
            td_surv = (td_surv_base * td_surv_step)
        else:
            td_surv_step = torch.rand((self.num_surv_samples, len(src))).to(self.device)
            td_surv = td_surv_step*(td_uv.view(1,-1))
            td_uv_non = td_surv_step * (td_uv.view(1, -1))

        if et is not None:
            ################# haweks
            # lambda_uv = self.hawkes_intensity(z_src, z_dst, et, td_pos)
            # lambda_surv = self.hawkes_intensity(u_non_embeddings, v_non_embeddings, et.repeat(self.num_surv_samples),
            #                                     td_surv).view(-1, len(src))
            ################# THP
            temporal_pos = (cur_time - last_time_pos) / (last_time_pos + 1)
            temporal_surv = (td_surv / (last_time_pos + 1).unsqueeze(0)).view(-1)
            lambda_uv = self.THP_intensity(z_src, z_dst, et, temporal_pos)
            lambda_surv = self.THP_intensity(u_non_embeddings, v_non_embeddings, et.repeat(self.num_surv_samples),
                                                temporal_surv).view(-1, len(src))

        else:
            ################# Hawkes
            # use the last_time to account for time
            # lambda_uv = self.hawkes_intensity_singleType(z_src, z_dst, td_uv, symmetric=False)
            # lambda_surv = self.hawkes_intensity_singleType(u_non_embeddings, v_non_embeddings, td_uv_non.view(-1),
            #                                             symmetric=False).view(-1, len(src))
            ################# decay
            # lambda_uv = self.decay_intensity_singleType(z_src, z_dst, torch.zeros(z_src.size(0), device=self.device),
            #                                             symmetric=False, out=self.decay_out)
            lambda_uv = self.decay_intensity_singleType(z_src, z_dst, last_uv,
                                                        symmetric=False, out=self.decay_out)
            lambda_surv = self.decay_intensity_singleType(u_non_embeddings, v_non_embeddings, td_uv_non.view(-1),
                                                        symmetric=False, out=self.decay_out).view(-1, len(src))
            ################# THP
            # temporal_pos = (cur_time - last_time_pos) / (last_time_pos + 1)
            # temporal_surv = (td_surv / (last_time_pos + 1).unsqueeze(0)).view(-1)
            # temporal_pos = (td_uv) / (t_uv + 1)
            # temporal_surv = (td_uv_non / (t_uv + 1).unsqueeze(0)).view(-1)
            #
            # lambda_uv = self.THP_intensity_singleType(z_src, z_dst, temporal_pos, symmetric=False)
            # lambda_surv = self.THP_intensity_singleType(u_non_embeddings, v_non_embeddings, temporal_surv,
            #                                             symmetric=False).view(-1, len(src))
        if accu_non_events is None:
            accu_non_events = torch.zeros((len(src),), device=self.device)
        if self.unit_sampling:
            integral = torch.sum(td_surv_step.view(1, -1) * lambda_surv, dim=0) + accu_non_events
            loss_surv = torch.sum(integral)
        else:
            integral = torch.mean(lambda_surv, dim=0) * td_pos + accu_non_events
            loss_surv = torch.sum(integral)

        loss_lambda = -torch.sum(torch.log(lambda_uv + 1e-7))

        cond_density = []
        if (not self.training) or (neg_dst is not None):
            with torch.no_grad():
                z_neg_dst = all_embeddings[assoc[neg_dst]]
                last_time_neg = torch.cat((last_update[assoc[src]].view(-1,1), last_update[assoc[neg_dst]].view(-1,1)), dim=1).max(-1)[0]
                td_neg = cur_time - last_time_neg
                if et is not None:
                    ############# Hawkes
                    # lambda_uv_neg = self.hawkes_intensity(z_src, z_neg_dst, et, td_neg)
                    ############# THP
                    temporal_neg = td_neg / (last_time_neg + 1)
                    lambda_uv_neg = self.THP_intensity(z_src, z_neg_dst, et, temporal_neg)
                else:
                    ############ Hawkes
                    # lambda_uv_neg = self.hawkes_intensity_singleType(z_src, z_neg_dst, td_neg, symmetric=False)
                    ############ THP
                    # temporal_neg = td_neg / (last_time_neg + 1)
                    # lambda_uv_neg = self.THP_intensity_singleType(z_src, z_neg_dst, temporal_neg, symmetric=False)
                    ############ decay
                    # lambda_uv_neg = self.decay_intensity_singleType(z_src, z_neg_dst,
                    #                                                 torch.zeros(z_src.size(0), device=self.device),
                    #                                                 symmetric=False, out=self.decay_out)
                    lambda_uv_neg = self.decay_intensity_singleType(z_src, z_neg_dst,
                                                                    td_neg, symmetric=False, out=self.decay_out)
                surv = torch.exp(-integral)
                assert len(z_src) == len(surv)
                cond_pos = lambda_uv #* surv
                cond_neg = lambda_uv_neg #* surv
                cond_density = [cond_pos, cond_neg]
                # cond_density = [lambda_uv, lambda_uv_neg]
        return loss_lambda/len(z_src), loss_surv/len(z_src), cond_density

    def decay_intensity_singleType(self, z_u, z_v, td, symmetric=True, out=False):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        if self.time_type == 'hr':
            td_norm = td / self.train_td_hr_max
        else:
            td_norm = td / self.train_td_max
            # td_norm = (td - train_td_mean) / train_td_std
        if symmetric:
            g_uv = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
            g_vu = self.omega[0](torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            g = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
        if out:
            # g_psi = g / (self.psi + 1e-7)
            g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)
            Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) * torch.exp(-self.w_t*td_norm)
        else:
            g = g - self.w_t*td_norm
            # g_psi = g / (self.psi + 1e-7)
            g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)
            Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi)
        return Lambda

    def hawkes_intensity_singleType(self, z_u, z_v, td, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        if self.time_type == 'hr':
            td_norm = td / self.train_td_hr_max
        else:
            td_norm = td / self.train_td_max
            # td_norm = (td - train_td_mean) / train_td_std
        if symmetric:
            g_uv = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
            g_vu = self.omega[0](torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            g = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
        # g_psi = g / (self.psi + 1e-7)
        g = g + self.alpha*torch.exp(-self.w_t*td_norm)
        g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * torch.log(1 + torch.exp(g_psi))
        Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) # + self.alpha*torch.exp(-self.w_t*td_norm)
        return Lambda

    def hawkes_intensity(self, z_u, z_v, et_uv, td, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        et = (et_uv > 0).long()
        if self.time_type == 'hr':
            td_norm = td / self.train_td_hr_max
            # td_norm = (td - self.train_td_hr_mean) / self.train_td_hr_std
        else:
            td_norm = td / self.train_td_max
            # td_norm = (td - train_td_mean) / train_td_std
        if symmetric:
            z_uv = torch.cat((z_u, z_v), dim=1)
            z_vu = torch.cat((z_v, z_u), dim=1)
            g_uv = z_uv.new_zeros(len(z_uv))
            g_vu = z_vu.new_zeros(len(z_vu))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g_uv[idx] = self.omega[k](z_uv).flatten()[idx]
                    g_vu[idx] = self.omega[k](z_vu).flatten()[idx]
                    # g_uv = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
            # g_vu = self.omega(torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            z_cat = torch.cat((z_u, z_v), dim=1)
            g = z_cat.new_zeros(len(z_cat))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g[idx] = self.omega[k](z_cat).flatten()[idx]

            # g = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
        psi = self.psi[et]
        alpha = self.alpha[et]
        w_t = self.w_t[et]
        # g += alpha*torch.exp(-w_t*td_norm)
        # g_psi = g / (psi + 1e-7)
        g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) #+ alpha*torch.exp(-w_t*td_norm)
        return Lambda

    def THP_intensity(self, z_u, z_v, et_uv, temporal, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        et = (et_uv > 0).long()

        if symmetric:
            z_uv = torch.cat((z_u, z_v), dim=1)
            z_vu = torch.cat((z_v, z_u), dim=1)
            g_uv = z_uv.new_zeros(len(z_uv))
            g_vu = z_vu.new_zeros(len(z_vu))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g_uv[idx] = self.omega[k](z_uv).flatten()[idx]
                    g_vu[idx] = self.omega[k](z_vu).flatten()[idx]
                    # g_uv = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
            # g_vu = self.omega(torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            z_cat = torch.cat((z_u, z_v), dim=1)
            g = z_cat.new_zeros(len(z_cat))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g[idx] = self.omega[k](z_cat).flatten()[idx]

            # g = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
        psi = self.psi[et]
        alpha = self.alpha[et]
        g += alpha*temporal
        # g_psi = g / (psi + 1e-7)
        g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) #+ alpha*torch.exp(-w_t*td_norm)
        return Lambda

    def THP_intensity_singleType(self, z_u, z_v, temporal, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        if symmetric:
            g_uv = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
            g_vu = self.omega[0](torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            g = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
        g = g + self.alpha * temporal
        # g_psi = g / (self.psi + 1e-7)
        g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * torch.log(1 + torch.exp(g_psi))
        Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) # + self.alpha*torch.exp(-self.w_t*td_norm)
        return Lambda

class DecoderTP_accu(torch.nn.Module):
    def __init__(self, embedding_dim, num_surv_samples, train_td_hr_max, train_td_max, start_t,
                 min_src_idx, max_src_idx, min_dst_idx, max_dst_idx,
                 train_td_hr_mean=None, train_td_hr_std=None, num_rel=2, time_type='hr', unit_sampling=True, device='cpu'):
        super(DecoderTP_accu, self).__init__()
        self.unit_sampling = unit_sampling
        self.embed_dim = embedding_dim
        self.num_surv_samples = num_surv_samples
        self.train_td_hr_max = train_td_hr_max

        self.train_td_max = train_td_max
        self.train_td_hr_mean = train_td_hr_mean
        self.train_td_hr_std = train_td_hr_std
        self.time_type = time_type
        self.device = device
        self.num_rel = num_rel
        self.min_src_idx = min_src_idx
        self.max_src_idx = max_src_idx
        self.min_dst_idx = min_dst_idx
        self.max_dst_idx = max_dst_idx

        self.omega = ModuleList([Linear(in_features=2*self.embed_dim, out_features=1) for _ in range(num_rel)])
        self.psi = Parameter(0.5*torch.ones(num_rel))
        self.alpha = Parameter(torch.rand(num_rel))
        # self.alpha = Parameter(-0.1*torch.ones(num_rel))
        self.w_t = Parameter(torch.rand(num_rel))

        z_last_memory = torch.empty(max_dst_idx-min_src_idx+1, embedding_dim, device=device)
        last_last_time = torch.zeros(max_dst_idx-min_src_idx+1, dtype=start_t.dtype, device=device)
        event_inten_accu = torch.empty(max_src_idx-min_src_idx+1, max_dst_idx-min_dst_idx+1, device=device)
        self.register_buffer('z_last_memory', z_last_memory)
        self.register_buffer('last_last_time', last_last_time)
        self.register_buffer('event_inten_accu', event_inten_accu)
        self.register_buffer('last_src', None)
        self.register_buffer('last_dst', None)

        self.last_src, self.last_dst = None, None

        self.reset_parameters()


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.z_last_memory)
        zeros(self.last_last_time)
        zeros(self.event_inten_accu)
        self.last_src, self.last_dst = None, None

    def forward(self, all_embeddings, assoc, src, pos_dst, last_update, cur_time,
                u_non_embeddings, v_non_embeddings, neg_dst=None, et=None, last_time_pos=None, update_accu=False):

        if update_accu:
            last_src_uni, last_dst_uni = self.last_src.unique(), self.last_dst.unique()
            update_src_neg = torch.tensor(np.delete(np.arange(self.min_src_idx, self.max_src_idx + 1), self.last_src.cpu().numpy() - self.min_src_idx),
                                          device=self.device)
            update_dst_neg = torch.tensor(np.delete(np.arange(self.min_dst_idx, self.max_dst_idx + 1), self.last_dst.cpu().numpy() - self.min_dst_idx),
                                          device=self.device)
            update_all_src = torch.cat([last_src_uni.view(-1, 1).repeat(1, update_dst_neg.size(0)).view(-1),
                              last_src_uni.view(-1, 1).repeat(1, last_dst_uni.size(0)).view(-1),
                              update_src_neg.repeat(last_dst_uni.size(0)).view(-1)])
            update_all_dst = torch.cat([update_dst_neg.repeat(last_src_uni.unique().size(0)).view(-1),
                                        last_dst_uni.repeat(last_src_uni.size(0)).view(-1),
                                        last_dst_uni.view(-1, 1).repeat(1, update_src_neg.size(0)).view(-1)])
            assert update_all_src.size(0)==update_all_dst.size(0)
            t1_start = last_update[assoc[last_src_uni]].view(-1, 1).repeat(1, update_dst_neg.size(0)).view(-1)
            t2_start = torch.cat([last_update[assoc[last_src_uni]].view(-1, 1).repeat(1, last_dst_uni.size(0)).view(-1,1),
                           last_update[assoc[last_dst_uni]].repeat(last_src_uni.size(0)).view(-1,1)], dim=1).max(-1)[0]
            t3_start = last_update[assoc[last_dst_uni]].view(-1, 1).repeat(1, update_src_neg.size(0)).view(-1)
            update_t_start = torch.cat([t1_start,t2_start,t3_start])
            t1_end = self.last_last_time[last_src_uni].view(-1, 1).repeat(1, update_dst_neg.size(0)).view(-1)
            t2_end = torch.cat([self.last_last_time[last_src_uni].view(-1, 1).repeat(1, last_dst_uni.size(0)).view(-1,1),
                           self.last_last_time[last_dst_uni].repeat(last_src_uni.size(0)).view(-1,1)], dim=1).max(-1)[0]
            t3_end = self.last_last_time[last_dst_uni].view(-1, 1).repeat(1, update_src_neg.size(0)).view(-1)
            update_t_end = torch.cat([t1_end, t2_end, t3_end])
            update_td = update_t_start - update_t_end

            self.event_inten_accu[update_all_src, update_all_dst-self.min_dst_idx] += (self.hawkes_intensity_singleType(
                self.z_last_memory[update_all_src], self.z_last_memory[update_all_dst], td=update_td) * update_td).detach()


        z_src, z_dst = all_embeddings[assoc[src]], all_embeddings[assoc[pos_dst]]

        if last_time_pos is None:
            last_time_pos = torch.cat((last_update[assoc[src]].view(-1,1),
                                       last_update[assoc[pos_dst]].view(-1,1)), dim=1).max(-1)[0]
            td_pos = cur_time - last_time_pos
        else:
            td_pos = cur_time - last_time_pos
            last_update_max = torch.cat((last_update[assoc[src]].view(-1,1),
                                       last_update[assoc[pos_dst]].view(-1,1)), dim=1).max(-1)[0]
            use_accu = last_time_pos >= last_update_max
            t_uv = torch.cat((last_update[assoc[src]].view(-1,1), last_update[assoc[pos_dst]].view(-1,1),
                              last_time_pos.view(-1,1)), dim=1).max(-1)[0]
            td_uv = cur_time - t_uv

        if self.unit_sampling:
            td_surv_step = td_pos / (self.num_surv_samples-1)
            td_surv_base = torch.linspace(0, self.num_surv_samples-1, self.num_surv_samples).unsqueeze(1) \
                .repeat(1, len(src)).to(self.device)
            td_surv = (td_surv_base * td_surv_step)
        else:
            td_surv_step = torch.rand((self.num_surv_samples, len(src))).to(self.device)
            td_surv = td_surv_step*(td_pos.view(1,-1))
            td_uv_non = td_surv_step*(td_uv.view(1,-1))

        if et is not None:
            ################# haweks
            # lambda_uv = self.hawkes_intensity(z_src, z_dst, et, td_pos)
            # lambda_surv = self.hawkes_intensity(u_non_embeddings, v_non_embeddings, et.repeat(self.num_surv_samples),
            #                                     td_surv).view(-1, len(src))
            ################# THP
            temporal_pos = (cur_time - last_time_pos) / (last_time_pos + 1)
            temporal_surv = (td_surv / (last_time_pos + 1).unsqueeze(0)).view(-1)
            lambda_uv = self.THP_intensity(z_src, z_dst, et, temporal_pos)
            lambda_surv = self.THP_intensity(u_non_embeddings, v_non_embeddings, et.repeat(self.num_surv_samples),
                                                temporal_surv).view(-1, len(src))

        else:
            ################# Hawkes
            # use the last_time to account for time
            lambda_uv = self.hawkes_intensity_singleType(z_src, z_dst, td_uv, symmetric=False)
            lambda_surv = self.hawkes_intensity_singleType(u_non_embeddings, v_non_embeddings, td_uv_non.view(-1),
                                                        symmetric=False).view(-1, len(src))
            ################# THP
            # temporal_pos = (cur_time - last_time_pos) / (last_time_pos + 1)
            # temporal_surv = (td_surv / (last_time_pos + 1).unsqueeze(0)).view(-1)
            # temporal_pos = (td_uv) / (t_uv + 1)
            # temporal_surv = (td_uv_non / (t_uv + 1).unsqueeze(0)).view(-1)
            #
            # lambda_uv = self.THP_intensity_singleType(z_src, z_dst, temporal_pos, symmetric=False)
            # lambda_surv = self.THP_intensity_singleType(u_non_embeddings, v_non_embeddings, temporal_surv,
            #                                             symmetric=False).view(-1, len(src))

        if self.unit_sampling:
            integral = torch.sum(td_surv_step.view(1, -1) * lambda_surv, dim=0)
            loss_surv = torch.sum(integral)
        else:
            integral = torch.mean(lambda_surv, dim=0) * td_uv + use_accu * self.event_inten_accu[src, pos_dst-self.min_dst_idx]
            loss_surv = torch.sum(integral)

        loss_lambda = -torch.sum(torch.log(lambda_uv + 1e-7))

        cond_density = []
        if (not self.training) or (neg_dst is not None):
            with torch.no_grad():
                z_neg_dst = all_embeddings[assoc[neg_dst]]
                last_time_neg = torch.cat((last_update[assoc[src]].view(-1,1), last_update[assoc[neg_dst]].view(-1,1)), dim=1).max(-1)[0]
                td_neg = cur_time - last_time_neg
                if et is not None:
                    ############# Hawkes
                    # lambda_uv_neg = self.hawkes_intensity(z_src, z_neg_dst, et, td_neg)
                    ############# THP
                    temporal_neg = td_neg / (last_time_neg + 1)
                    lambda_uv_neg = self.THP_intensity(z_src, z_neg_dst, et, temporal_neg)
                else:
                    ############ Hawkes
                    # lambda_uv_neg = self.hawkes_intensity_singleType(z_src, z_neg_dst, td_neg, symmetric=False)
                    ############ THP
                    temporal_neg = td_neg / (last_time_neg + 1)
                    lambda_uv_neg = self.THP_intensity_singleType(z_src, z_neg_dst, temporal_neg, symmetric=False)
                surv = torch.exp(-integral)
                assert len(z_src) == len(surv)
                cond_pos = lambda_uv * surv
                cond_neg = lambda_uv_neg * surv
                cond_density = [cond_pos, cond_neg]
                # cond_density = [lambda_uv, lambda_uv_neg]

        if update_accu:
            self.z_last_memory[src] = all_embeddings[assoc[src]].clone()
            self.z_last_memory[pos_dst] = all_embeddings[assoc[pos_dst]].clone()
        else:
            self.z_last_memory[:,:] = all_embeddings.detach().clone()

        self.last_last_time[src] = last_update[assoc[src]].clone()
        self.last_last_time[pos_dst] = last_update[assoc[pos_dst]].clone()

        self.last_src = src.clone()
        self.last_dst = pos_dst.clone()

        self.event_inten_accu[src,pos_dst-self.min_dst_idx] = 0.

        return loss_lambda/len(z_src), loss_surv/len(z_src), cond_density

    def hawkes_intensity_singleType(self, z_u, z_v, td, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        if self.time_type == 'hr':
            td_norm = td / self.train_td_hr_max
        else:
            td_norm = td / self.train_td_max
            # td_norm = (td - train_td_mean) / train_td_std
        if symmetric:
            g_uv = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
            g_vu = self.omega[0](torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            g = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
        # g_psi = g / (self.psi + 1e-7)
        g = g + self.alpha*torch.exp(-self.w_t*td_norm)
        g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * torch.log(1 + torch.exp(g_psi))
        Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) # + self.alpha*torch.exp(-self.w_t*td_norm)
        return Lambda

    def hawkes_intensity(self, z_u, z_v, et_uv, td, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        et = (et_uv > 0).long()
        if self.time_type == 'hr':
            td_norm = td / self.train_td_hr_max
            # td_norm = (td - self.train_td_hr_mean) / self.train_td_hr_std
        else:
            td_norm = td / self.train_td_max
            # td_norm = (td - train_td_mean) / train_td_std
        if symmetric:
            z_uv = torch.cat((z_u, z_v), dim=1)
            z_vu = torch.cat((z_v, z_u), dim=1)
            g_uv = z_uv.new_zeros(len(z_uv))
            g_vu = z_vu.new_zeros(len(z_vu))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g_uv[idx] = self.omega[k](z_uv).flatten()[idx]
                    g_vu[idx] = self.omega[k](z_vu).flatten()[idx]
                    # g_uv = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
            # g_vu = self.omega(torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            z_cat = torch.cat((z_u, z_v), dim=1)
            g = z_cat.new_zeros(len(z_cat))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g[idx] = self.omega[k](z_cat).flatten()[idx]

            # g = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
        psi = self.psi[et]
        alpha = self.alpha[et]
        w_t = self.w_t[et]
        # g += alpha*torch.exp(-w_t*td_norm)
        # g_psi = g / (psi + 1e-7)
        g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) #+ alpha*torch.exp(-w_t*td_norm)
        return Lambda

    def THP_intensity(self, z_u, z_v, et_uv, temporal, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        et = (et_uv > 0).long()

        if symmetric:
            z_uv = torch.cat((z_u, z_v), dim=1)
            z_vu = torch.cat((z_v, z_u), dim=1)
            g_uv = z_uv.new_zeros(len(z_uv))
            g_vu = z_vu.new_zeros(len(z_vu))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g_uv[idx] = self.omega[k](z_uv).flatten()[idx]
                    g_vu[idx] = self.omega[k](z_vu).flatten()[idx]
                    # g_uv = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
            # g_vu = self.omega(torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            z_cat = torch.cat((z_u, z_v), dim=1)
            g = z_cat.new_zeros(len(z_cat))
            for k in range(2):
                idx = (et == k)
                if torch.sum(idx) > 0:
                    g[idx] = self.omega[k](z_cat).flatten()[idx]

            # g = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
        psi = self.psi[et]
        alpha = self.alpha[et]
        g += alpha*temporal
        # g_psi = g / (psi + 1e-7)
        g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) #+ alpha*torch.exp(-w_t*td_norm)
        return Lambda

    def THP_intensity_singleType(self, z_u, z_v, temporal, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        if symmetric:
            g_uv = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
            g_vu = self.omega[0](torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            g = self.omega[0](torch.cat((z_u, z_v), dim=1)).flatten()
        g = g + self.alpha * temporal
        # g_psi = g / (self.psi + 1e-7)
        g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * torch.log(1 + torch.exp(g_psi))
        Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) # + self.alpha*torch.exp(-self.w_t*td_norm)
        return Lambda