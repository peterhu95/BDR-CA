import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB, EmbAttentionLayer
from copy import deepcopy

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions_ingroup = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]
        self.attentions_outgroup = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions_ingroup):
            self.add_module('attention_ingroup{}'.format(i), attention)
        for i, attention in enumerate(self.attentions_outgroup):
            self.add_module('attention_outgroup{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )
        
        self.emb_att = EmbAttentionLayer( nhid * nheads, nhid,dropout=dropout, alpha=alpha)
        

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        edge_embed_nhop = relation_embed[
            edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        x_s1 = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions_ingroup], dim=1)
        x_s1 = self.dropout_layer(x_s1)
        
        edge_list_over = deepcopy(edge_list)
        edge_list_over[0,:],edge_list_over[1,:] = deepcopy((edge_list_over[1,:],edge_list_over[0,:]))
        
        edge_list_nhop_over = deepcopy(edge_list_nhop)
        edge_list_nhop_over[0,:],edge_list_nhop_over[1,:] = deepcopy((edge_list_nhop_over[1,:],edge_list_nhop_over[0,:]))
        
        x_s2 = torch.cat([att(x, edge_list_over, edge_embed, edge_list_nhop_over, edge_embed_nhop)
                       for att in self.attentions_outgroup], dim=1)
        x_s2 = self.dropout_layer(x_s2)
        

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_nhop = out_relation_1[
            edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        x_s1 = F.elu(self.out_att(x_s1, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        x_s2 = F.elu(self.out_att(x_s2, edge_list_over, edge_embed,
                               edge_list_nhop_over, edge_embed_nhop))
        
        # x = x_s1 + x_s2  # average
        x = self.emb_att(x_s1, x_s2)  # 1 att
                               
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]

        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels1, conv_out_channels2):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels1 = conv_out_channels1
        self.conv_out_channels2 = conv_out_channels2

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 2, 1,
                             self.conv_out_channels2, self.drop_conv, self.alpha_conv)
                             
        self.convAuto = ConvAutoencoder(self.entity_out_dim_1 * self.nheads_GAT_1, 2, 1, self.conv_out_channels1, self.drop_conv)
        
        self.BN = torch.nn.BatchNorm1d(2)

    def forward(self, pos_neg_ratio, Corpus_, adj, batch_inputs):
        #conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
        #    batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        batch_size, length = batch_inputs.size()
        positive_bs = int(batch_size/(pos_neg_ratio+1))
        
        conv_input_ht = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        ht_rep, conv_input_ht_prime = self.convAuto(self.BN(conv_input_ht))  # 3 dimensions
        
        r = self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1)
        logits = torch.matmul(ht_rep,r.transpose(1,2))
        
        #positive_logits = -torch.log(torch.sigmoid(torch.matmul(ht_rep[:positive_bs,:,:], r[:positive_bs,:,:].transpose(1,2))))  # - because minimize loss
        #negative_logits = -torch.log(torch.sigmoid(torch.matmul(torch.neg(ht_rep[positive_bs:,:,:]), r[positive_bs:,:,:].transpose(1,2))))
        #out_conv = self.convKB(conv_input_rep_r)
        return logits.view(-1,1), conv_input_ht, conv_input_ht_prime

    def batch_test(self, batch_inputs):
        #conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
        #    batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        
        
        conv_input_ht = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        ht_rep, _ = self.convAuto(self.BN(conv_input_ht))  # 3 dimensions
        
        r = self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1)
        out_conv = torch.sigmoid(torch.matmul(ht_rep, r.transpose(1,2)))
        #out_conv = self.convKB(conv_input_rep_r)
        
        return out_conv.view(-1, 1)


class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        #self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        #nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        
        self.encoder = nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len)),
        nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(
            out_channels, in_channels, (1, input_seq_len)),
        nn.LeakyReLU(True)
        )
        
        self.W_laten1 = nn.Parameter(torch.zeros(
            size=(out_channels*input_dim, input_dim)))  # (out_channels*input_dim, input_dim)
        nn.init.xavier_uniform_(self.W_laten1.data, gain=1.414)
        #self.W_laten2 = nn.Parameter(torch.zeros(
        #    size=(input_dim, out_channels*input_dim)))  # (input_dim, out_channels*input_dim)
        #nn.init.xavier_uniform_(self.W_laten2.data, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        laten = self.encoder(conv_input)
        ht_rep = laten.squeeze(-1).view(batch_size, -1)  # (b, 10000)
        ht_rep = ht_rep.mm(self.W_laten1)  # (b, 200) 
        ht_rep = ht_rep.unsqueeze(1)  # (b, 1, 200)
        
        ht_rep_prime = ht_rep.squeeze(1)
        ht_rep_prime = ht_rep_prime.mm(self.W_laten1.t())
        ht_rep_prime = ht_rep_prime.reshape(batch_size, -1, dim, 1)
        
        conv_input_prime = self.decoder(ht_rep_prime)
        conv_input_prime = conv_input_prime.squeeze(1).transpose(1,2)
        
        #out_conv = self.dropout(
        #    self.non_linearity(self.conv_layer(conv_input)))

        
        return ht_rep, conv_input_prime

