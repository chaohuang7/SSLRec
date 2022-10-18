from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict, calcRegLoss
import faiss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(max(args.gnn_layer, args.hyperLayer * 2))])
		self.kmeans = KMeans()
	
	def getEgoEmbeds(self, adj):
		uEmbeds, iEmbeds, _ = self.forward(adj)
		return t.concat([uEmbeds, iEmbeds], axis=0)

	def eStep(self):
		self.usrCentroids, self.usr2Clust, usrClustNums = self.kmeans(self.uEmbeds.detach())
		self.itmCentroids, self.itm2Clust, itmClustNums = self.kmeans(self.iEmbeds.detach())
		print((usrClustNums == 0).sum(), (itmClustNums == 0).sum())

	def runKMeans(self, x):
		kmeans = faiss.Kmeans(d=args.latdim, k=args.k, gpu=True)
		kmeans.train(x)
		clusterCents = kmeans.centroids
		_, I = kmeans.index.search(x, 1)

		centroids = t.Tensor(clusterCents).cuda()
		node2clust = t.LongTensor(I).squeeze().cuda()
		return centroids, node2clust

	def forward(self, adj):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst[:args.gnn_layer+1])# / len(embedsLst)
		return embeds[:args.user], embeds[args.user:], embedsLst
	
	def calcBPRLoss(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		return bprLoss
	
	def infoNCE(self, embeds1, embeds2, nodes1, nodes2, temp=args.temp):
		embeds1 = F.normalize(embeds1 + 1e-12, p=2)
		embeds2 = F.normalize(embeds2 + 1e-12, p=2)
		pckEmbeds1 = embeds1[nodes1]
		pckEmbeds2 = embeds2[nodes2]
		nume = t.exp((pckEmbeds1 * pckEmbeds2).sum(-1) / temp)
		deno = t.exp((pckEmbeds1 @ embeds2.T / temp)).sum(-1) + 1e-12
		sslLoss = -t.log(nume / deno + 1e-12).mean()
		return sslLoss
	
	def calcStructLoss(self, ctxEmbeds, egoEmbeds, usrs, itms):
		uEmbeds1, iEmbeds1 = t.split(ctxEmbeds, [args.user, args.item])
		uEmbeds2, iEmbeds2 = t.split(egoEmbeds, [args.user, args.item])
		return self.infoNCE(uEmbeds1, uEmbeds2, usrs, usrs) + self.infoNCE(iEmbeds1, iEmbeds2, itms, itms)
	
	def calcProtoLoss(self, egoEmbeds, usrs, itms):
		uEmbeds, iEmbeds = t.split(egoEmbeds, [args.user, args.item])
		usrClust = self.usr2Clust[usrs]
		itmClust = self.itm2Clust[itms]
		return self.infoNCE(uEmbeds, self.usrCentroids, usrs, usrClust) + self.infoNCE(iEmbeds, self.itmCentroids, itms, itmClust)
	
	def calcRegLoss(self, egoEmbeds, ancs, poss, negs):
		uEmbeds, iEmbeds = t.split(egoEmbeds, [args.user, args.item])
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		ret = ancEmbeds.norm(2).square() + posEmbeds.norm(2).square() + negEmbeds.norm(2).square()
		return ret
	
	def calcLoss(self, adj, ancs, poss, negs):
		uEmbeds, iEmbeds, embedsLst = self.forward(adj)
		egoEmbeds = embedsLst[0]
		ctxEmbeds = embedsLst[args.hyperLayer * 2]
		
		structLoss = self.calcStructLoss(ctxEmbeds, egoEmbeds, ancs, poss) * args.structReg
		protoLoss = self.calcProtoLoss(egoEmbeds, ancs, poss) * args.protoReg
		bprLoss = self.calcBPRLoss(uEmbeds, iEmbeds, ancs, poss, negs)
		regLoss = self.calcRegLoss(egoEmbeds, ancs, poss, negs) * args.reg
		# regLoss = calcRegLoss(self) * args.reg
		return bprLoss, protoLoss, structLoss, regLoss

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)

class KMeans(nn.Module):
	def __init__(self):
		super(KMeans, self).__init__()
	
	def forward(self, x):
		centroids = t.rand([args.k, args.latdim]).cuda()
		ones = t.ones([x.shape[0], 1]).cuda()
		for i in range(1000):
			dists = (x.view([-1, 1, args.latdim]) - centroids.view([1, -1, args.latdim])).square().sum(-1)
			_, idxs = t.min(dists, dim=1)
			newCents = t.zeros_like(centroids)
			newCents.index_add_(0, idxs, x)
			clustNums = t.zeros([centroids.shape[0], 1]).cuda()
			clustNums.index_add_(0, idxs, ones)
			centroids = newCents / (clustNums + 1e-6)
		return centroids, idxs, clustNums
