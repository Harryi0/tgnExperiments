# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realsitic and a better
# test bed for future methods.

import os.path as osp
from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from datetime import datetime, timedelta
from collections import defaultdict
from torch.nn import Linear, Parameter, ModuleList, Sequential, Softplus
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from torch_geometric.datasets import JODIEDataset
# from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn import TransformerConv
from tgnMemoryNoMsg import TGNMemory

# from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage,
#                                            LastAggregator)
from tgnMemoryNoMsg import (LastNeighborLoader, IdentityMessage,
                                           LastAggregator)

from SocialEvolutionDataset import SocialEvolutionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')
dataset = SocialEvolutionDataset(path, name='social_initial')
data = dataset[0].to(device)

num_nodes = max(data.src.max(), data.dst.max())+1
num_links = len(data.y.unique())

train_len = 44106
# train_len = 43834
# only comm 43469

# Train len 43834, test len 10535
# Train len 44106, test len 10535

bipartite = False

# Ensure to only sample actual destination nodes as negatives.
# min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
# min_src_idx, max_src_idx = int(data.src.min()), int(data.src.max())

train_data, test_data = data[:train_len], data[train_len:]

# train_data, val_data, test_data = data.train_val_test_split(
#     val_ratio=0.15, test_ratio=0.15)
train_data_small = data[:20]

def normalize_td(data):
    dic = defaultdict(list)
    src, dst, t = data.src.cpu().numpy(), data.dst.cpu().numpy(), data.t.cpu().numpy()
    for i in range(len(src)):
        dic[src[i]].append(t[i])
        dic[dst[i]].append(t[i])
    all_diff = []
    all_td = []
    for k, v in dic.items():
        ts = np.array(v)
        td = np.diff(ts)
        all_diff.append(td)
        if len(v) >= 2:
            timestamp = np.array(list(map(lambda x: datetime.fromtimestamp(x), v)))
            delta = np.diff(timestamp)
            all_td.append(delta)
    all_diff = np.concatenate(all_diff)
    all_td = np.concatenate(all_td)
    all_td_hr = np.array(list(map(lambda x: round(x.days * 24 + x.seconds / 3600, 3), all_td)))
    return all_diff.mean(), all_diff.std(), all_diff.max(), \
           round(all_td_hr.mean(),3), round(all_td_hr.std(),3), round(all_td_hr.max(), 3)

train_td_mean, train_td_std, train_td_max, train_td_hr_mean, train_td_hr_std, train_td_hr_max = normalize_td(train_data)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

def get_return_time_multitype(dataset, train=False):
    reoccur_dict = {}
    dataset_src, dataset_dst, dataset_t = dataset.src.cpu().numpy(), dataset.dst.cpu().numpy(), dataset.t.cpu().numpy()
    dataset_et = dataset.y.cpu().numpy()
    for i in range(len(dataset_src)):
        n1, n2, t, et = dataset_src[i], dataset_dst[i], dataset_t[i], dataset_et[i]
        # et = 0 if r in data_set.assoc_types else 1
        key = (n1,n2,et) if n1<=n2 else (n2,n1,et)
        if key not in reoccur_dict:
            reoccur_dict[key] = [t]
        elif t == reoccur_dict[key][-1]:
            continue
        else:
            reoccur_dict[key].append(t)
    count = 0
    for _, occ in reoccur_dict.items():
        if len(occ) > 1:
            count += len(occ)-1
    print("Number of repeat events in the data : {}/{}".format(count, len(dataset_src)))
    if train:
        end_time = datetime(2009, 5, 1)
    else:
        end_time = datetime(2009, 6, 30)
    reoccur_time_ts = np.zeros(len(dataset_src))
    reoccur_time_hr = np.zeros(len(dataset_src))
    reoccur_time_hr_true = np.ones(len(dataset_src))*(-1)
    for idx in range(len(dataset_src)):
        n1, n2, t, et = dataset_src[idx], dataset_dst[idx], dataset_t[idx], dataset_et[idx]
        key = (n1,n2,et) if n1<=n2 else (n2,n1,et)
        occ = reoccur_dict[key]
        if len(occ) == 1 or t == occ[-1]:
            reoccur_time_ts[idx] = end_time.timestamp()
            reoccur_time = end_time - datetime.fromtimestamp(int(t))
            reoccur_time_hr[idx] = round((reoccur_time.days * 24 + reoccur_time.seconds / 3600), 3)
        else:
            reoccur_time_ts[idx] = occ[occ.index(t) + 1]
            reoccur_time = datetime.fromtimestamp(int(reoccur_time_ts[idx])) - datetime.fromtimestamp(int(t))
            reoccur_time_hr[idx] = round((reoccur_time.days * 24 + reoccur_time.seconds / 3600), 3)
            reoccur_time_hr_true[idx] = round((reoccur_time.days * 24 + reoccur_time.seconds / 3600), 3)

    return reoccur_dict, reoccur_time_ts, reoccur_time_hr, reoccur_time_hr_true


def get_return_time(dataset):
    reoccur_dict = {}
    dataset_src, dataset_dst, dataset_t = dataset.src.cpu().numpy(), dataset.dst.cpu().numpy(), dataset.t.cpu().numpy()
    for i in range(len(dataset_src)):
        n1, n2, t = dataset_src[i], dataset_dst[i], dataset_t[i]
        key = (n1, n2)
        if key not in reoccur_dict:
            reoccur_dict[key] = [t]
        elif t == reoccur_dict[key][-1]:
            continue
        else:
            reoccur_dict[key].append(t)
    count = 0
    for _, occ in reoccur_dict.items():
        if len(occ) > 1:
            count += len(occ)-1
    print("Number of repeat events in the data : {}/{}".format(count, len(dataset_src)))
    end_time = dataset_t[-1]+1
    reoccur_time_ts = np.zeros(len(dataset_src))
    reoccur_time_hr = np.zeros(len(dataset_src))
    for idx in range(len(dataset_src)):
        n1, n2, t = dataset_src[idx], dataset_dst[idx], dataset_t[idx]
        occ = reoccur_dict[(n1,n2)]
        if len(occ) == 1 or t == occ[-1]:
            reoccur_time_ts[idx] = end_time
            reoccur_time = datetime.fromtimestamp(int(end_time)) - datetime.fromtimestamp(int(t))
            reoccur_time_hr[idx] = round((reoccur_time.days*24 + reoccur_time.seconds/3600),3)
        else:
            reoccur_time_ts[idx] = occ[occ.index(t) + 1]
            reoccur_time = datetime.fromtimestamp(int(reoccur_time_ts[idx])) - datetime.fromtimestamp(int(t))
            reoccur_time_hr[idx] = round((reoccur_time.days*24 + reoccur_time.seconds/3600),3)

    return reoccur_dict, reoccur_time_ts, reoccur_time_hr

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        # edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        edge_attr = rel_t_enc
        return self.conv(x, edge_index, edge_attr)

class HawkesLSTMCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, args):
        super(HawkesLSTMCell, self).__init__()
        self.input_g = Linear(input_dim + hidden_size, hidden_size)
        self.forget_g = Linear(input_dim + hidden_size, hidden_size)
        self.output_g = Linear(input_dim + hidden_size, hidden_size)
        self.input_target = Linear(input_dim + hidden_size, hidden_size)
        self.forget_target = Linear(input_dim + hidden_size, hidden_size)
        self.z_gate = Linear(input_dim + hidden_size, hidden_size)
        self.decay_layer = Sequential(
            Linear(input_dim + hidden_size, hidden_size),
            Softplus(beta=args.softrelu_scale))

    def forward(self, x, h_t, c_t, c_target):
        """
        Compute the updated LSTM paramters.

        Args:s
            x: (ent_emb, rel_emb, aggregated_emb)
            h_t: cont. hidden state at timestamp t
            c_t: cont. cell state at timestamp t
            c_target: target cell state

        Returns:
            h_i: just-updated hidden state
            h_t: hidden state just before next event
            cell_i: just-updated cell state
            c_t: cell state decayed to before next event
            c_target_i: cell state target before the next event
            output: LSTM output
            decay_i: rate of decay for the cell state
        """
        v = torch.cat((x, h_t), dim=1)
        inpt = torch.sigmoid(self.input_g(v))
        forget = torch.sigmoid(self.forget_g(v))
        input_target = torch.sigmoid(self.input_target(v))
        forget_target = torch.sigmoid(self.forget_target(v))
        output = torch.sigmoid(self.output_g(v))  # compute output gate
        # Not-quite-c
        z_i = torch.tanh(self.z_gate(v))
        # Compute the decay parameter
        decay = self.decay_layer(v)
        # Update the cell state to c(t_i+)
        c_i = forget * c_t + inpt * z_i
        # Update the cell state target
        c_target = forget_target * c_target + input_target * z_i

        return c_i, c_target, output, decay

class DyRepDecoder(torch.nn.Module):
    def __init__(self, embedding_dim, num_surv_samples):
        super(DyRepDecoder, self).__init__()
        self.embed_dim = embedding_dim
        self.num_surv_samples = num_surv_samples
        self.omega = ModuleList([Linear(in_features=2*self.embed_dim, out_features=1),
                                Linear(in_features=2 * self.embed_dim, out_features=1)])

        self.symmetric = True

        self.psi = Parameter(0.5*torch.ones(2))
        self.alpha = Parameter(torch.rand(2))
        self.w_t = Parameter(torch.rand(2))
        self.reset_parameters()


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def forward(self, all_embeddings, assoc, src, pos_dst, neg_dst_surv,
                neg_src_surv=None, neg_dst=None, last_update=None, cur_time=None, et=None):

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

        z_neg_dst = all_embeddings[assoc[neg_dst]]
        last_time_neg = torch.cat((last_update[assoc[src]].view(-1, 1), last_update[assoc[neg_dst]].view(-1, 1)), dim=1).max(-1)[0]
        td_neg = cur_time - last_time_neg
        lambda_uv_neg = self.hawkes_intensity(z_src, z_neg_dst, et, td_neg)
        s = surv_u.view(-1, self.num_surv_samples).mean(dim=-1) + surv_v.view(-1, self.num_surv_samples).mean(dim=-1)
        surv = torch.exp(-s)
        cond_pos = lambda_uv * surv
        cond_neg = lambda_uv_neg * surv

        all_td = torch.linspace(0, h_max, num_samples).unsqueeze(1).repeat(1, len(src)).view(-1)

        embeddings_u_tp = z_src.repeat(num_samples, 1)
        embeddings_v_tp = z_dst.repeat(num_samples, 1)

        if self.symmetric:
            intensity = 0.5 * (
                    self.hawkes_intensity(embeddings_u_tp, embeddings_v_tp, et.repeat(num_samples), all_td)
                    .view(-1, len(src)) +
                    self.hawkes_intensity(embeddings_u_tp, embeddings_v_tp, et.repeat(num_samples), all_td)
                    .view(-1, len(src)))
        else:
            intensity = self.hawkes_intensity(embeddings_u_tp, embeddings_v_tp, et.repeat(num_samples),
                                               all_td) \
                .view(-1, len(src))

        integral = torch.cumsum(timestep * intensity, dim=0)
        density = (intensity * torch.exp(-integral))
        t_sample = all_td.view(-1, len(src)) * density
        return_time_pred = (timestep * 0.5 * (t_sample[:-1] + t_sample[1:])).sum(dim=0)

        return cond_pos, cond_neg, return_time_pred

    def hawkes_intensity(self, z_u, z_v, et_uv, td, symmetric=True):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        et = (et_uv > 0).long()
        # td_norm = td
        td_norm = td / train_td_max
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
        g_psi = g / (psi + 1e-7)
        # g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + alpha*torch.exp(-w_t*td_norm)
        return Lambda

class Decoder(torch.nn.Module):
    def __init__(self, embedding_dim, num_nodes):
        super(Decoder, self).__init__()
        self.embed_dim = embedding_dim
        self.num_nodes = num_nodes
        self.omega = ModuleList([Linear(in_features=2*self.embed_dim, out_features=1),
                                Linear(in_features=2 * self.embed_dim, out_features=1)])

        self.psi = Parameter(0.5*torch.ones(2))
        self.alpha = Parameter(torch.rand(2))
        self.w_t = Parameter(torch.rand(2))

        self.symmetric = True

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def forward(self, all_embeddings, assoc, src, pos_dst, neg_dst, last_update, cur_time, et):

        z_src, z_dst = all_embeddings[assoc[src]], all_embeddings[assoc[pos_dst]]
        z_dst_neg = all_embeddings[assoc[neg_dst]]

        # last_time_pos = torch.cat((last_update[assoc[src]].view(-1, 1),
        #                            last_update[assoc[pos_dst]].view(-1, 1)), dim=1).max(-1)[0]
        # last_time_neg = torch.cat((last_update[assoc[src]].view(-1, 1),
        #                            last_update[assoc[neg_dst]].view(-1, 1)), dim=1).max(-1)[0]
        # td_pos = cur_time - last_time_pos
        # td_neg = cur_time - last_time_neg
        # lambda_pos = self.hawkes_intensity(z_src, z_dst, et, td_pos)
        # lambda_neg = self.hawkes_intensity(z_src, z_dst_neg, et, td_neg)

        td_src = cur_time - last_update[assoc[src]]
        td_dst = cur_time - last_update[assoc[pos_dst]]
        lambda_src = self.hawkes_intensity(
            z_src.unsqueeze(1).repeat(1, self.num_nodes,1).view(-1, self.embed_dim),
            all_embeddings.repeat(len(src),1),
            et.unsqueeze(1).repeat(1, self.num_nodes).view(-1), td_src.unsqueeze(1).repeat(1, self.num_nodes).view(-1)
        ).view(-1, self.num_nodes)
        lambda_dst = self.hawkes_intensity(
            all_embeddings.repeat(len(src), 1),
            z_dst.unsqueeze(1).repeat(1, self.num_nodes,1).view(-1, self.embed_dim),
            et.unsqueeze(1).repeat(1, self.num_nodes).view(-1), td_dst.unsqueeze(1).repeat(1, self.num_nodes).view(-1)
        ).view(-1, self.num_nodes)



        all_td = torch.linspace(0, h_max, num_samples).unsqueeze(1).repeat(1, len(src)).view(-1)

        embeddings_u_tp = z_src.repeat(num_samples, 1)
        embeddings_v_tp = z_dst.repeat(num_samples, 1)

        if self.symmetric:
            intensity = 0.5 * (
                    self.hawkes_intensity(embeddings_u_tp, embeddings_v_tp, et.repeat(num_samples), all_td)
                    .view(-1, len(src)) +
                    self.hawkes_intensity(embeddings_u_tp, embeddings_v_tp, et.repeat(num_samples), all_td)
                    .view(-1, len(src)))
        else:
            intensity = self.hawkes_intensity(embeddings_u_tp, embeddings_v_tp, et.repeat(num_samples),
                                               all_td) \
                .view(-1, len(src))

        integral = torch.cumsum(timestep * intensity, dim=0)
        density = (intensity * torch.exp(-integral))
        t_sample = all_td.view(-1, len(src)) * density
        return_time_pred = (timestep * 0.5 * (t_sample[:-1] + t_sample[1:])).sum(dim=0)

        return lambda_src, lambda_dst, return_time_pred
        # return lambda_pos, lambda_neg, return_time_pred

    def hawkes_intensity(self, z_u, z_v, et_uv, td, symmetric=False):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        et = (et_uv > 0).long()
        # td_norm = td
        td_norm = td / train_td_max
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
        g_psi = g / (psi + 1e-7)
        # g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) #+ alpha*torch.exp(-w_t*td_norm)
        return Lambda


memory_dim = time_dim = embedding_dim = 32
link_dim = 1

memory = TGNMemory(
    data.num_nodes,
    0,
    memory_dim,
    time_dim,
    message_module=IdentityMessage(0, memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=0,
    time_enc=memory.time_enc,
).to(device)

num_surv_samples = 20
num_time_samples = 5

dyrep = Decoder(
    embedding_dim=embedding_dim,
    num_nodes=num_nodes
).to(device)

# dyrep = DyRepDecoder(
#     embedding_dim=embedding_dim,
#     num_surv_samples=num_surv_samples
# ).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(dyrep.parameters()), lr=0.0001)

# optimizer = torch.optim.Adam(
#     set(memory.parameters()) | set(gnn.parameters()), lr=0.001)


link_criterion = torch.nn.BCEWithLogitsLoss()
link_criterion_multi = torch.nn.CrossEntropyLoss()
time_criterion = torch.nn.MSELoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


# Get return time for val and test dataset

# val_reoccur_dict, val_return_ts, val_return_hr = get_return_time(val_data)
# test_reoccur_dict, test_return_ts, test_return_hr = get_return_time(test_data)

_, _, train_return_hr, train_return_hr_true = get_return_time_multitype(train_data, train=True)

_, _, test_return_hr, test_return_hr_true = get_return_time_multitype(test_data)

h_max = 5000
timestep = 1

num_samples = int(h_max / timestep) + 1

first_batch = []

tp_scale = 1e-6


def time_pred(batch_src, batch_pos_dst, batch_t, batch_link_type, batch_last_update, batch_assoc, batch_z, random_state,
              batch_all_neg_nodes):
    return_time_pred = []
    # making time prediction  for each node in the batch
    with torch.no_grad():
        for src_c, pos_dst_c, t_c, et_c in zip(batch_src, batch_pos_dst, batch_t, batch_link_type):
            # just update the current node to the memory
            # memory.update_state(src_c.expand(2), pos_dst_c.expand(2), t_c.expand(2), msg_c.view(1,-1).expand(2,-1))

            t_cur_date = datetime.fromtimestamp(int(t_c))
            # Take the most recent last update time in the node pair
            t_prev = datetime.fromtimestamp(int(max(batch_last_update[batch_assoc[src_c]], batch_last_update[batch_assoc[pos_dst_c]])))
            # The time difference between current time and most recent update time would be a base for the future time sampling
            td = t_cur_date - t_prev
            time_scale_hour = round((td.days * 24 + td.seconds / 3600), 3)

            # random generate factor [0,2] for the time sampling
            factor_samples = 2 * random_state.rand(num_time_samples)
            sampled_time_scale = time_scale_hour * factor_samples

            embeddings_u = batch_z[batch_assoc[src_c]].expand(num_time_samples, -1)
            embeddings_v = batch_z[batch_assoc[pos_dst_c]].expand(num_time_samples, -1)

            t_c_n = torch.tensor(list(map(lambda x: int((t_cur_date + timedelta(hours=x)).timestamp()),
                                          np.cumsum(sampled_time_scale))), device=device)
            all_td_c = t_c_n - t_c

            all_neg_sample = random_state.choice(
                batch_all_neg_nodes,
                size=num_surv_samples * 2 * num_time_samples,
                replace=len(batch_all_neg_nodes) < num_surv_samples * 2 * num_time_samples)
            neg_src_c = all_neg_sample[:num_surv_samples * num_time_samples]
            neg_dst_c = all_neg_sample[num_surv_samples * num_time_samples:]

            embeddings_u_neg = torch.cat((
                batch_z[batch_assoc[src_c]].view(1, -1).expand(num_surv_samples * num_time_samples, -1),
                batch_z[batch_assoc[neg_dst_c]]), dim=0)
            embeddings_v_neg = torch.cat((
                batch_z[batch_assoc[neg_src_c]],
                batch_z[batch_assoc[pos_dst_c]].view(1, -1).expand(num_surv_samples * num_time_samples, -1)), dim=0)

            all_td_c_expand = all_td_c.unsqueeze(1).repeat(1, num_surv_samples).view(-1, 1).repeat(2, 1).view(-1)
            # all_td_c_expand = all_td_c.unsqueeze(1).repeat(1, num_surv_samples).view(-1)
            surv_0 = dyrep.hawkes_intensity(embeddings_u_neg, embeddings_v_neg,
                                            torch.zeros(len(embeddings_u_neg)), all_td_c_expand)
            # surv_0 = 0
            surv_1 = dyrep.hawkes_intensity(embeddings_u_neg, embeddings_v_neg,
                                            torch.ones(len(embeddings_u_neg)), all_td_c_expand)

            intensity = (surv_0 + surv_1).view(-1, num_surv_samples).mean(dim=-1)
            surv_allsamples = intensity[:num_time_samples] + intensity[num_time_samples:]
            lambda_t_allsamples = dyrep.hawkes_intensity(embeddings_u, embeddings_v, et_c.repeat(num_time_samples),
                                                         all_td_c)
            f_samples = lambda_t_allsamples * torch.exp(-surv_allsamples)
            # expectation = ((torch.from_numpy(np.cumsum(sampled_time_scale)).to(
            #     device) - train_td_hr_mean) / train_td_hr_std) * f_samples
            expectation = torch.from_numpy(np.cumsum(sampled_time_scale)) * f_samples

            return_time_pred.append(expectation.sum() / num_time_samples)
        return_time_pred = torch.stack(return_time_pred)
    return return_time_pred


def time_pred_unitsample(batch_src, batch_pos_dst, batch_link_type, batch_z, batch_assoc, symmetric=True):
    with torch.no_grad():
        num_samples = int(h_max / timestep) + 1
        all_td = torch.linspace(0, h_max, num_samples).unsqueeze(1).repeat(1, len(batch_src)).view(-1)

        embeddings_u = batch_z[batch_assoc[batch_src]].repeat(num_samples, 1)
        embeddings_v = batch_z[batch_assoc[batch_pos_dst]].repeat(num_samples, 1)

        if symmetric:
            intensity = 0.5 * (
                    dyrep.hawkes_intensity(embeddings_u, embeddings_v, batch_link_type.repeat(num_samples), all_td)
                    .view(-1, len(batch_src)) +
                    dyrep.hawkes_intensity(embeddings_v, embeddings_u, batch_link_type.repeat(num_samples), all_td)
                    .view(-1, len(batch_src)))
        else:
            intensity = dyrep.hawkes_intensity(embeddings_u, embeddings_v, batch_link_type.repeat(num_samples), all_td)\
                .view(-1, len(batch_src))

        integral = torch.cumsum(timestep * intensity, dim=0)
        density = (intensity * torch.exp(-integral))
        t_sample = all_td.view(-1, len(batch_src)) * density
        return_time_pred = (timestep * 0.5 * (t_sample[:-1] + t_sample[1:])).sum(dim=0)
    return return_time_pred


def train(dataset,return_time_hr, batch_size=200, total_batches=220, time_prediction=False, link_prediction=False):
    dataset_len = dataset.num_events
    return_time_hr = torch.tensor(return_time_hr, dtype=float).to(device)

    memory.train()
    gnn.train()
    dyrep.train()

    random_state = np.random.RandomState(12345)
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    # link_embedding.reset_parameters()

    total_loss = 0.
    total_lp_loss, total_tp_loss = 0., 0.

    total_mae = 0.
    all_aps = []

    for batch_id, batch in enumerate(tqdm(dataset.seq_batches(batch_size=batch_size), total=total_batches)):
        optimizer.zero_grad()

        src, pos_dst, t, link_type = batch.src, batch.dst, batch.t, batch.y

        all_neg_nodes = np.delete(np.arange(num_nodes), np.concatenate([pos_dst.cpu().numpy(), src.cpu().numpy()]))

        # all_neg_surv = torch.tensor(random_state.choice(all_neg_nodes, size=src.size(0)*num_surv_samples*2,
        #                                         replace=len(all_neg_nodes) < src.size(0)*num_surv_samples*2),
        #                             device=device)
        # neg_dst_surv = all_neg_surv[:src.size(0)*num_surv_samples]
        # neg_src_surv = all_neg_surv[src.size(0)*num_surv_samples:]


        neg_dst = torch.tensor(random_state.choice(all_neg_nodes, size=src.size(0),
                                                replace=len(all_neg_nodes) < src.size(0)),device=device)

        # n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        # n_id = torch.cat([src, pos_dst, neg_dst, neg_dst_surv, neg_src_surv]).unique()
        n_id = torch.arange(num_nodes, device=device)

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], None) #link_embedding(data.y[e_id]).detach()

        lambda_src, lambda_dst, tp = dyrep(z, assoc, src, pos_dst, neg_dst, last_update, t, link_type)
        lp_loss =link_criterion_multi(lambda_src, pos_dst)
        lp_loss += link_criterion_multi(lambda_dst, src)

        # lambda_pos, lambda_neg, tp = dyrep(z, assoc, src, pos_dst, neg_dst, last_update, t, link_type)
        # lp_loss = link_criterion(lambda_pos, torch.ones_like(lambda_pos))
        # lp_loss += link_criterion(lambda_neg, torch.zeros_like(lambda_neg))
        ############# olp dyrep method
        # cond_pos, cond_neg, tp = dyrep(z, assoc, src, pos_dst, neg_dst_surv,
        #                                neg_src_surv=neg_src_surv, last_update=last_update, cur_time=t, neg_dst=neg_dst,
        #                                et=link_type)
        # lp_loss = link_criterion(cond_pos, torch.ones_like(cond_pos))
        # lp_loss += link_criterion(cond_neg, torch.zeros_like(cond_neg))


        batch_return_time = return_time_hr[batch_id * batch_size:(batch_id * batch_size + batch.num_events)]
        # tp_loss = 0
        tp_loss = tp_scale*torch.square(tp-batch_return_time).mean()
        # tp_loss = time_criterion(tp, batch_return_time)

        loss = lp_loss + tp_loss

        mae, ap = 0, 0
        # if link_prediction:
        #     with torch.no_grad():
        #         y_pred = torch.cat([lambda_pos, lambda_neg], dim=0).sigmoid().cpu()
        #         y_true = torch.cat([torch.ones(lambda_pos.size(0)),torch.zeros(lambda_neg.size(0))], dim=0)
        #         ap = average_precision_score(y_true, y_pred)
        #         all_aps.append(ap)

        if time_prediction:
            with torch.no_grad():
                mae = np.mean(abs((tp - batch_return_time).cpu().numpy()))
                total_mae += mae*len(batch.src)

        if (batch_id) % 50 == 0:
            if batch_id==0:
                first_batch.append(loss)
            print("Batch {}, Loss {}, Link pred loss {}, Time pred loss {}, link prediction ap {}, "
                  "time prediction mae {}".format(
                batch_id+1, loss, lp_loss, tp_loss, ap, mae))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, None) #link_embedding(link_type).detach()
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        # torch.nn.utils.clip_grad_value_(dyrep.parameters(), 100)
        # torch.nn.utils.clip_grad_value_(gnn.parameters(), 50)
        # torch.nn.utils.clip_grad_value_(memory.parameters(), 50)
        optimizer.step()

        # dyrep.psi.data = torch.clamp(dyrep.psi.data, 1e-1, 1e+3)
        memory.detach()
        total_loss += float(loss) * batch.num_events
        total_lp_loss += float(lp_loss) * batch.num_events
        total_tp_loss += float(tp_loss) * batch.num_events

        # if batch_id > 20:
        #     break

    return total_loss/dataset_len, total_lp_loss/dataset_len, \
           total_tp_loss/dataset_len, float(torch.tensor(all_aps).mean()), total_mae/dataset_len


@torch.no_grad()
def test(inference_data, return_time_hr, batch_size=200, total_batches=53):
    memory.eval()
    gnn.eval()
    dyrep.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.
    random_state = np.random.RandomState(12345)

    return_time_hr = torch.tensor(return_time_hr).to(device)

    total_loss = 0
    total_lp_loss, total_tp_loss = 0, 0
    total_mae = 0
    ap = 0
    all_aps = []
    for batch_id, batch in enumerate(tqdm(inference_data.seq_batches(batch_size=batch_size), total=total_batches)):
        src, pos_dst, t, link_type = batch.src, batch.dst, batch.t, batch.y

        all_neg_nodes = np.delete(np.arange(num_nodes), np.concatenate([pos_dst.cpu().numpy(), src.cpu().numpy()]))

        neg_dst = torch.tensor(random_state.choice(all_neg_nodes, size=src.size(0),
                                                replace=len(all_neg_nodes) < src.size(0)),
                               device=device)
        ####### include all dst nodes
        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        # n_id = torch.arange(num_nodes)

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], None)

        assert z.size(0) > max(assoc[n_id])

        lambda_src, lambda_dst, tp = dyrep(z, assoc, src, pos_dst, neg_dst, last_update, t, link_type)
        lp_loss =link_criterion_multi(lambda_src, pos_dst)
        lp_loss += link_criterion_multi(lambda_dst, src)

        # lambda_pos, lambda_neg, tp = dyrep(z, assoc, src, pos_dst, neg_dst, last_update, t, link_type)
        # lp_loss = link_criterion(lambda_pos, torch.ones_like(lambda_pos))
        # lp_loss += link_criterion(lambda_neg, torch.zeros_like(lambda_neg))


        batch_return_time = return_time_hr[batch_id * batch_size:(batch_id * batch_size + batch.num_events)]
        tp_loss = tp_scale*time_criterion(tp, batch_return_time)

        loss = lp_loss + tp_loss

        total_loss += float(loss) * batch.num_events

        # y_pred = torch.cat([lambda_pos, lambda_neg], dim=0).sigmoid().cpu()
        # y_true = torch.cat([torch.ones(lambda_pos.size(0)), torch.zeros(lambda_neg.size(0))], dim=0)
        # ap = average_precision_score(y_true, y_pred)
        # all_aps.append(ap)

        mae = np.mean(abs((tp - batch_return_time).cpu().numpy()))
        total_mae += mae * len(batch.src)


        memory.update_state(src, pos_dst, t, None) #link_embedding(link_type).detach()
        neighbor_loader.insert(src, pos_dst)

        if batch_id % 20 == 0:
            print("Test Batch {}, loss {}, link pred ap {}, time pred MAE {}"
                  .format(batch_id+1, loss, ap, mae))


    return total_loss/inference_data.num_events, float(torch.tensor(all_aps).mean()), \
           total_mae/inference_data.num_events

all_loss, all_loss_lp, all_loss_tp  = [], [], []
all_train_ap, all_train_mae =  [], []
all_val_loss, all_test_loss = [], []
all_val_ap, all_val_auc, all_test_ap, all_test_auc = [], [], [], []
all_val_mae, all_test_mae = [], []
epochs = 5
epochs_no_improve = 0
patience = 20
early_stop = False
min_test_mae = float('inf')
for epoch in range(1, epochs+1): #51
    # , return_time_hr=train_return_hr, time_prediction=False
    loss, loss_lp, loss_tp, train_ap, train_mae = train(train_data, return_time_hr=train_return_hr,
                                                        time_prediction=True, link_prediction=True)#, batch_size=5, total_batches=4
    all_loss.append(loss)
    all_loss_lp.append(loss_lp)
    all_loss_tp.append(loss_tp)
    all_train_ap.append(train_ap)
    all_train_mae.append(train_mae)
    print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}, loss_lp:{loss_lp:.4f}, '
          f'loss_tp:{loss_tp:.4f}, train_ap:{train_ap: .4f}, train mae {train_mae:.4f}')
    test_loss, test_ap, test_mae = test(test_data, test_return_hr)
    print(f' Epoch: {epoch:02d}, Val AP: {test_ap:.4f} , Val LOSS: {test_loss:.4f}, Val MAE: {test_mae:.4f}')
    all_test_ap.append(test_ap)
    all_test_loss.append(test_loss)
    all_test_mae.append(test_mae)
    if early_stop:
        if test_mae < min_test_mae:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
            min_test_mae = test_mae


    # val_ap, val_auc, val_loss, val_mae = test(val_data, val_return_hr)
    # print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}, Val LOSS: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
    # all_val_ap.append(val_ap)
    # all_val_auc.append(val_auc)
    # all_val_loss.append(val_loss)
    # all_val_mae.append(val_mae)

fig = plt.figure(figsize=(18, 5))
ax = plt.subplot(1,3,1)
plt.plot(np.arange(1, len(all_loss)+1), np.array(all_loss), 'k', label='total loss')
plt.plot(np.arange(1, len(all_loss_lp)+1), np.array(all_loss_lp), 'r', label='loss lp')
plt.plot(np.arange(1, len(all_loss_tp)+1), np.array(all_loss_tp), 'b', label='loss tp')
plt.legend()
plt.title("train loss")
plt.subplot(1,3,2)
plt.plot(np.arange(1,len(all_train_ap)+1), np.array(all_train_ap), 'r')
plt.title("train link pred ap")
plt.subplot(1,3,3)
plt.plot(np.arange(1,len(all_train_mae)+1), np.array(all_train_mae), 'b')
plt.title("train time prediction mae")
fig.savefig('tgnNoMsg_GHNN_train.png')

fig2 = plt.figure(figsize=(18, 5))
plt.subplot(1,3,1)
plt.plot(np.arange(1, len(all_test_loss)+1), np.array(all_test_loss), 'k', label='total loss')
plt.title("test loss")
plt.subplot(1,3,2)
plt.plot(np.arange(1, len(all_test_ap)+1), np.array(all_test_ap), 'r', label='total loss')
plt.title("test ap")
plt.subplot(1,3,3)
plt.plot(np.arange(1, len(all_test_mae)+1), np.array(all_test_mae), 'b', label='total loss')
plt.title("test mae")
fig2.savefig('tgnNoMsg_GHNN_test.png')
# fig.savefig('tgnHawkes_20events_twoSurv_lr1e-3.png')
# fig2.savefig('tgnHawkesFirstBatch_allData_twoSurv_lr1e-3.png')
    # val_ap, val_auc = test(val_data, val_return_hr)
    # test_ap, test_auc = test(test_data, test_return_hr)
    # print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
    # print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')