import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.conv_layer2 = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)
        #self.Wtrans = nn.Linear((input_dim) * out_channels, input_dim)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer2.weight, gain=1.414)
        #nn.init.xavier_uniform_(self.Wtrans.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        
        conv_input_tail = conv_input[:, 2, :].unsqueeze(1).transpose(1, 2).unsqueeze(1)   # tail
        conv_input = torch.cat((conv_input[:, 0, :].unsqueeze(1), conv_input[:, 1, :].unsqueeze(1)), dim=1).transpose(1, 2).unsqueeze(1)  # head | rel
        
        # assuming inputs are of the form ->
        #conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        #conv_input = conv_input.unsqueeze(1)

        out_conv = self.non_linearity(self.conv_layer(conv_input))
        
        #tail_prime = self.Wtrans(out_conv.squeeze(-1).view(batch_size, -1)).unsqueeze(1).transpose(1, 2).unsqueeze(1)
        

        conv_input = torch.cat((out_conv, conv_input_tail), dim=3)
        
        out_conv = self.dropout(
            self.non_linearity(self.conv_layer2(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        N = input.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
            

class EmbAttentionLayer(nn.Module):
    """
    Embedding attention
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(EmbAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, in_features)))
        nn.init.xavier_normal_(self.a_1.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, emb1, emb2):
        N = emb1.size()[0]

        emb = torch.cat((emb1, emb2), dim=0).t()  # (d, 2*N)
        
        emb_m = self.a_1.mm(emb)

        powers = -self.leakyrelu(self.a_2.mm(emb_m))  # (1, 2*N)
        
        weights = torch.softmax(torch.cat((powers[0,0:N].unsqueeze(0), powers[0,N:2*N].unsqueeze(0)), dim=0), dim=0)  # (2, N)
        
        assert not torch.isnan(weights).any()

        weights = torch.cat((weights[0,:].unsqueeze(0), weights[1,:].unsqueeze(0)),dim=1)  # (1, 2*N)
        weights = weights.repeat(self.in_features, 1)  # (d, 2*N)
        emb = emb*weights                            # (d, 2*N)
        
        emb = (emb[:,0:N]+emb[:,N:2*N]).t()  # (d, N)

        emb = self.dropout(emb)
        return emb

    
