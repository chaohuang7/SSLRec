from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
		self.perturbGcnLayers1 = nn.Sequential(*[GCNLayer(perturb=True) for i in range(args.gnn_layer)])
		self.perturbGcnLayers2 = nn.Sequential(*[GCNLayer(perturb=True) for i in range(args.gnn_layer)])

	
	def getEgoEmbeds(self, adj):
		uEmbeds, iEmbeds = self.forward(adj)
		return t.concat([uEmbeds, iEmbeds], axis=0)

	def forward(self, adj):
		iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)# / len(embedsLst)

		if self.training:
			perturbEmbedsLst1 = [iniEmbeds]
			for gcn in self.perturbGcnLayers1:
				embeds = gcn(adj, perturbEmbedsLst1[-1])
				perturbEmbedsLst1.append(embeds)
			perturbEmbeds1 = sum(perturbEmbedsLst1)# / len(embedsLst)

			perturbEmbedsLst2 = [iniEmbeds]
			for gcn in self.perturbGcnLayers2:
				embeds = gcn(adj, perturbEmbedsLst2[-1])
				perturbEmbedsLst2.append(embeds)
			perturbEmbeds2 = sum(perturbEmbedsLst2)# / len(embedsLst)

			return mainEmbeds[:args.user], mainEmbeds[args.user:], perturbEmbeds1[:args.user], perturbEmbeds1[args.user:], perturbEmbeds2[:args.user], perturbEmbeds2[args.user:]
		return mainEmbeds[:args.user], mainEmbeds[args.user:]

class GCNLayer(nn.Module):
	def __init__(self, perturb=False):
		super(GCNLayer, self).__init__()
		self.perturb = perturb

	def forward(self, adj, embeds):
		ret = t.spmm(adj, embeds)
		if not self.perturb:
			return ret
		noise = (F.normalize(t.rand(ret.shape).cuda(), p=2) * t.sign(ret)) * args.eps
		return ret + noise
