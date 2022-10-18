import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	# ret += (model.usrStruct + model.itmStruct)
	return ret

# def calcReward(bprLoss, keepRate):
# 	# return t.where(bprLoss >= threshold, 1.0, xi)
# 	_, posLocs = t.topk(bprLoss, int(bprLoss.shape[0] * (1 - keepRate)))
# 	ones = t.ones_like(bprLoss).cuda()
# 	reward = t.minimum(bprLoss, ones * (0.5 - 1e-6))
# 	pckBprLoss = bprLoss[posLocs]
# 	ones = t.ones_like(pckBprLoss).cuda()
# 	reward[posLocs] = t.minimum(t.maximum(pckBprLoss, ones * (0.5 + 1e-6)), ones)
# 	return reward

def calcReward(bprLossDiff, keepRate):
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	reward = t.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	ret = 0
	for p in model.parameters():
		if p.grad is not None:
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def contrast(embeds1, embeds2, nodes, temp):
	uniqNodes = t.unique(nodes)
	pckEmbeds1 = F.normalize(embeds1[uniqNodes], p=2)
	pckEmbeds2 = F.normalize(embeds2[uniqNodes], p=2)
	posScores = (pckEmbeds1 * pckEmbeds2).sum(-1) / temp
	negScores = (pckEmbeds1 @ pckEmbeds2.T) / temp
	clLoss = -(t.log(t.exp(posScores) / t.exp(negScores).sum(-1))).mean()
	return clLoss