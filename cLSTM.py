import torch
from torch.nn import Linear, Sequential, Softplus
from torch_geometric.nn.inits import zeros


class cLSTMCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(cLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.i = Linear(input_dim+hidden_dim, hidden_dim)
        self.i_target = Linear(input_dim+hidden_dim, hidden_dim)
        self.f = Linear(input_dim+hidden_dim, hidden_dim)
        self.f_target = Linear(input_dim+hidden_dim, hidden_dim)
        self.z = Linear(input_dim+hidden_dim, hidden_dim)
        self.o = Linear(input_dim+hidden_dim, hidden_dim)
        self.delta = Linear(input_dim+hidden_dim, hidden_dim)
        self.delta_f = Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def forward(self, input, hidden, c, c_target):
        k = torch.cat([input, hidden], dim=1)

        gate_i = torch.sigmoid(self.i(k))
        gate_i_target = torch.sigmoid(self.i_target(k))
        gate_f = torch.sigmoid(self.f(k))
        gate_f_target = torch.sigmoid(self.f_target(k))
        gate_o = torch.sigmoid(self.o(k))
        z = 2 * torch.sigmoid(self.z(k)) - 1
        c_n = gate_f * c + gate_i * z
        c_target_n = gate_f_target * c_target + gate_i_target * z
        delta = self.delta_f(self.delta(k))

        return c_n, c_target_n, gate_o, delta


class cLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_nodes=None):
        super(cLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.clstm = cLSTMCell(2*input_dim, hidden_dim)

        # self.register_buffer('hidden_memory', torch.empty(num_nodes, hidden_dim))
        # self.register_buffer('cell_memory', torch.empty(num_nodes, hidden_dim))
        # self.register_buffer('cell_target_memory', torch.empty(num_nodes, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.clstm.reset_parameters()
        # self.reset_state()

    def reset_state(self):
        zeros(self.hidden_memory)
        zeros(self.cell_memory)
        zeros(self.cell_target_memory)

    def initialization(self, batch_size):
        h0 = torch.zeros((batch_size, self.hidden_dim), device=self.device)
        c0 = torch.zeros((batch_size, self.hidden_dim), device=self.device)
        c_target0 = torch.zeros((batch_size, self.hidden_dim), device=self.device)
        return h0, c0, c_target0

    def forward(self, x, n_id, t, last_t, x_neighbor=None, x_type=None):
        h, c, c_target = self.initialization(x.size(0))
        # h_last = self.hidden_memory[n_id]
        # c = self.cell_memory[n_id]
        # c_target = self.cell_target_memory[n_id]
        if x_neighbor is not None:
            x = torch.cat([x_neighbor, x], dim=1)
        if x_type is not None:
            x = torch.cat([x, x_type], dim=1)
        delta_t = t-last_t
        c_n, c_target_n, gate_o, delta = self.clstm(x, h, c, c_target)
        cell = c_target_n + (c_n - c_target_n) * torch.exp(-delta*(delta_t.view(-1,1)))
        h_n = gate_o * torch.tanh(cell)
        return h_n, c_n, c_target_n, gate_o, delta
        # samples = z[edge_index[0]]
        # labels = edge_index[1]
        # labels = labels.view(-1, 1).expand(-1, samples.size(1))
        # unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        # assoc_l = torch.empty_like(assoc, dtype=torch.long)
        # assoc_l[labels.unique()] = torch.arange(labels.unique().size(0))
        # labels_assoc = assoc_l[labels]
        # labels_assoc = labels_assoc.view(-1, 1).expand(-1, samples.size(1))
        # unique_labels, labels_count = labels_assoc.unique(dim=0, return_counts=True)
        # res = torch.zeros((n_id.size(0), z.size(1)), dtype=torch.float).scatter_add_(0, labels, samples)
        #
        # res[labels.unique()] /= labels_count.float().unsqueeze(1)
        # pass