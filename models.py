from torch_geometric.nn import GATConv, GINConv
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch
import sklearn
import sklearn.cluster

class GCN(nn.Module):
    '''
    2-layer GCN with dropout
    '''
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
def cluster(data, k, temp, num_iter, init = None, cluster_temp=5):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    '''
    #normalize x so it lies on the unit sphere
    data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    #use kmeans++ initialization if nothing is provided
    if init is None:
        data_np = data.detach().numpy()
        norm = (data_np**2).sum(axis=1)
        init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        init = torch.tensor(init, requires_grad=True)
        if num_iter == 0: return init
    mu = init
    n = data.shape[0]
    d = data.shape[1]
#    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
    for t in range(num_iter):
        #get distances between all data points and cluster centers
#        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
        dist = data @ mu.t()
        #cluster responsibilities via softmax
        r = torch.softmax(cluster_temp*dist, 1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist

class GCNClusterNet(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    '''
    def __init__(self, nfeat, nhid, nout, dropout, K, cluster_temp):
        super(GCNClusterNet, self).__init__()

        self.GCN = GCN(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, x, adj, num_iter=1):
        embeds = self.GCN(x, adj)
        mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist = cluster(embeds, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        return mu, r, embeds, dist


class GINGeom(nn.Module):
    '''
    2-layer GCN with dropout
    '''
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GINGeom, self).__init__()

        self.gc1 = GINConv(torch.nn.Linear(nfeat, nhid))
        self.gc2 = GINConv(torch.nn.Linear(nhid, nout))
        self.dropout = dropout

    def forward(self, x, adj):
        # print("INSIDE GINGEOM")
        edge_index = adj.indices()
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        # print("X\n\n",x)
        return x


class GINClusterNet(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    '''
    def __init__(self, nfeat, nhid, nout, dropout, K, cluster_temp):
        super(GINClusterNet, self).__init__()

        self.GIN = GINGeom(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, x, adj, num_iter=1):
        # print("INSIDE FORWARD")
        embeds = self.GIN(x, adj)
        # print("EMBEDS\n\n",embeds)
        mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist = cluster(embeds, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        # print("PRINTING")
        # print("mu\n\n",mu)
        # print("r\n\n",r)
        # print("embeds\n\n",embeds)
        # print("dist\n\n",dist)
        return mu, r, embeds, dist


class GCNDeep(nn.Module):
    '''
    A stack of nlayers GCNs. The first maps nfeat -> nhid features, the 
    middle layers all map nhid -> nhid, and the last maps nhid -> nout.
    '''
    def __init__(self, nfeat, nhid, nout, dropout, nlayers):
        super(GCNDeep, self).__init__()

        self.gcstart = GraphConvolution(nfeat, nhid)
        self.gcn_middle = []
        for i in range(nlayers-2):
            self.gcn_middle.append(GraphConvolution(nhid, nhid))
        self.gcend = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcstart(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for gc in self.gcn_middle:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcend(x, adj)

        return x
    

class GCNDeepSigmoid(nn.Module):
    '''
    Deep archicture that, instead of being intended to output a cluster membership
    for each node (as in GCNDeep), outputs instead a probability distribution over
    the nodes. Used for problems like facility location where the goal is to select
    a subset of K nodes. 
    '''
    def __init__(self, nfeat, nhid, nout, dropout, nlayers):
        super(GCNDeepSigmoid, self).__init__()

        self.gcstart = GraphConvolution(nfeat, nhid)
        self.gcn_middle = []
        for i in range(nlayers-2):
            self.gcn_middle.append(GraphConvolution(nhid, nhid))
        self.gcend = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcstart(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for gc in self.gcn_middle:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcend(x, adj)
        x = torch.nn.Sigmoid()(x).flatten()
        return x


    
class GCNLink(nn.Module):
    '''
    GCN link prediction model based on:
    
    M. Schlichtkrull, T. Kipf, P. Bloem, R. Van Den Berg, I. Titov, and M. Welling. Modeling
    416 relational data with graph convolutional networks. In European Semantic Web Conference,
    417 2018.
    '''
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCNLink, self).__init__()

        self.GCN = GCN(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, adj, to_pred):
        embeds = self.GCN(x, adj)
        dot = (embeds[to_pred[:, 0]]*self.distmult.expand(to_pred.shape[0], self.distmult.shape[0])*embeds[to_pred[:, 1]]).sum(dim=1)
        return dot


class GATGeom(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GATGeom, self).__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(nfeat, nhid, dropout=dropout, heads=nheads, negative_slope=alpha)
        self.conv2 = GATConv(nhid*nheads, nclass, dropout=dropout, heads=1, negative_slope=alpha)

    def forward(self, x, adj):
        edge_index = adj.indices()
        x = F.dropout(x, p=self.dropout, training=self.training)
#        print(x)
#        print(torch.isnan(x).any())
        x = F.elu(self.conv1(x, edge_index))
#        print(x)
#        print(torch.isnan(x).any())
        x = F.dropout(x, p=self.dropout, training=self.training)
#        print(x)
#        print(torch.isnan(x).any())
        x = self.conv2(x, edge_index)
#        print(x)
#        print(torch.isnan(x).any())
        x = x + 0.000001
        return x


class GCNClusterGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, K, cluster_temp, alpha = 0.2, nheads = 2):
        super(GCNClusterGAT, self).__init__()

        self.GAT = GATGeom(nfeat, nhid, nout, dropout, alpha, nheads)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, x, adj, num_iter=1):
        embeds = self.GAT(x, adj)
        mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist = cluster(embeds, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        return mu, r, embeds, dist




