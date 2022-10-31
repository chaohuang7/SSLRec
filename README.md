# Self-Supervised Learning for Recommendation

<b>SSLRec</b> is a PyTorch-based deep learning framework for recommender systems enhanced by self-supervised learning techniques. It contains commonly-used datasets, code scripts for data processing, training, testing, evaluation, and state-of-the-art research models.

## Implemented Methods
| Model | Paper | Venue |
|:-----:|:-----|:-----:|
|LightGCN| <a href='https://dl.acm.org/doi/pdf/10.1145/3397271.3401063' target='_blank'>Lightgcn: Simplifying and powering graph convolution network for recommendation</a> | SIGIR'20|
|SGL| <a href='https://dl.acm.org/doi/pdf/10.1145/3404835.3462862' target='_blank'>Self-supervised graph learning for recommendation</a>| SIGIR'21 |
|HCCF|<a href='https://dl.acm.org/doi/pdf/10.1145/3477495.3532058' target='_blank'>Hypergraph Contrastive Collaborative Filtering</a> | SIGIR'22 |
|NCL| <a href='https://dl.acm.org/doi/pdf/10.1145/3485447.3512104' targets='_blank'>Improving graph collaborative filtering with neighborhood-enriched contrastive learning</a>| WWW'22|
|SimGCL| <a href='https://dl.acm.org/doi/pdf/10.1145/3477495.3531937' targets='_blank'>Are graph augmentations necessary? simple graph contrastive learning for recommendation</a>| SIGIR'22|

## Environment
<b>SSLRec</b> is implemented under the following development environment:
* python==3.10.4
* numpy==1.22.3
* torch==1.11.0
* scipy=1.7.3

## Datasets
| Dataset | \# Users | \# Items | \# Interactions | Interaction Density |
|:-------:|:--------:|:--------:|:---------------:|:-------:|
|Sparse Gowalla|$25,557$|$19,747$|$294,983$|$5.9\times 10^{-4}$|
|Sparse Yelp   |$42,712$|$26,822$|$182,357$|$1.6\times 10^{-4}$|
|Sparse Amazon |$76,469$|$83,761$|$966,680$|$1.5\times 10^{-4}$|
|Yelp | $29,601$|$24,734$|$1,517,326$|$2.1\times 10^{-3}$|
|MovieLens| $69,878$ |$10,196$|$9,988,816$|$1.4\times 10^{-2}$|
|Amazon|$78,578$|$77,801$|$3,190,224$|$5.2\times 10^{-4}$|

## Usage
To run a specific methods, change your directory to the corresponding directory (e.g. methods/sgl/). Run the training and testing with this command line: `python Main.py`. Some important arguments shared by most methods are as follows:
* `data`: This is a string arguments specifying which dataset to run on. Currently we have released six datasets: `sp_gowalla`, `sp_yelp`, `sp_amazon`, `yelp`, `ml10m` and `amazon`.
* `latdim`: This denotes the dimensionality of latent embeddings. By default it is set as `32` for all methods.
* `gnn_layer`: For GNN-based methods, this parameter determines the number of GNN layers.
* `reg`: This parameter determines the weight for weight-decay regularization ($l_2$ regularization).
* `ssl_reg`: For most methods that utilize only one SSL training objective, this parameter specify the weight for SSL loss term.
* `temp`: For most methods that utilize only one SSL training objective, this parameter specify the temperature factor for SSL.
* `keepRate`: For methods that conduct random drop, this parameter specify the rate to keep values.
