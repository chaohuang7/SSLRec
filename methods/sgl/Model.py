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

		self.edgeDropper = SpAdjDropEdge(args.keepRate)

	def getEgoEmbeds(self, adj):
		uEmbeds, iEmbeds = self.forward(adj)
		return t.concat([uEmbeds, iEmbeds], axis=0)

	def forward(self, adj, keepRate=1.0):
		iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		
		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)# / len(embedsLst)

		if keepRate == 1.0:
			return mainEmbeds[:args.user], mainEmbeds[args.user:]

		# for edge drop
		if args.aug_data == 'ed' or args.aug_data == 'ED':
			adjView1 = self.edgeDropper(adj, keepRate)
			embedsLst = [iniEmbeds]
			for gcn in self.gcnLayers:
				embeds = gcn(adjView1, embedsLst[-1])
				embedsLst.append(embeds)
			embedsView1 = sum(embedsLst)

			adjView2 = self.edgeDropper(adj, keepRate)
			embedsLst = [iniEmbeds]
			for gcn in self.gcnLayers:
				embeds = gcn(adjView2, embedsLst[-1])
				embedsLst.append(embeds)
			embedsView2 = sum(embedsLst)
		# for random walk
		elif args.aug_data == 'rw' or args.aug_data == 'RW':
			embedsLst = [iniEmbeds]
			for gcn in self.gcnLayers:
				temadj = self.edgeDropper(adj, keepRate)
				embeds = gcn(temadj, embedsLst[-1])
				embedsLst.append(embeds)
			embedsView1 = sum(embedsLst)

			embedsLst = [iniEmbeds]
			for gcn in self.gcnLayers:
				temadj = self.edgeDropper(adj, keepRate)
				embeds = gcn(temadj, embedsLst[-1])
				embedsLst.append(embeds)
			embedsView2 = sum(embedsLst)
		# for node drop
		elif args.aug_data == 'nd' or args.aug_data == 'ND':
			rdmMask = (t.rand(iniEmbeds.shape[0]) < keepRate) * 1.0
			embedsLst = [iniEmbeds]
			for gcn in self.gcnLayers:
				embeds = gcn(adj, embedsLst[-1] * rdmMask)
				embedsLst.append(embeds)
			embedsView1 = sum(embedsLst)

			rdmMask = (t.rand(iniEmbeds.shape[0]) < keepRate) * 1.0
			embedsLst = [iniEmbeds]
			for gcn in self.gcnLayers:
				embeds = gcn(adj, embedsLst[-1] * rdmMask)
				embedsLst.append(embeds)
			embedsView2 = sum(embedsLst)
		return mainEmbeds[:args.user], mainEmbeds[args.user:], embedsView1[:args.user], embedsView1[args.user:], embedsView2[:args.user], embedsView2[args.user:]

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
