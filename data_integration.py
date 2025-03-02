# Source code: https://spatialglue-tutorials.readthedocs.io/en/latest/Tutorial%202_data%20integration%20for%20mouse%20thymus%20Stereo-CITE-seq.html
import os
import torch
import pandas as pd
import scanpy as sc
import SpatialGlue
from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.preprocess import clr_normalize_each_cell, pca


file_fold =   "/home/fenosoa/scratch/mouse_thymus_Stereo-CITE-seq/Mouse_Thymus/"

adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# the location of R, which is required for the 'mclust' clustering algorithm. 
os.environ['R_HOME'] = '/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Core/r/4.3.1/lib64/R'


# Specify data type
data_type = 'Stereo-CITE-seq'
# Fix random seed
from SpatialGlue.preprocess import fix_seed
random_seed = 2022
fix_seed(random_seed)


# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.filter_cells(adata_omics1, min_genes=80)

sc.pp.filter_genes(adata_omics2, min_cells=50)
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()

sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)

# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)

# Constructing neighbor graph
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

# Training the model
# define model
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
model = Train_SpatialGlue(data, datatype=data_type, device=device)

# train model
output = model.train()

adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1']
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2']
adata.obsm['SpatialGlue'] = output['SpatialGlue']
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']

# Cross-omics integrative analysis
# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from SpatialGlue.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=8, method=tool, use_pca=True)

# visualization
import matplotlib.pyplot as plt
adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]

fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=30)
sc.tl.umap(adata)

sc.pl.umap(adata, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=20, show=False)

plt.tight_layout(w_pad=0.3)
plt.savefig("sRNA_vs_protein.png", dpi=300, bbox_inches="tight")
plt.savefig("sRNA_vs_protein.tiff", dpi=3000, bbox_inches="tight")
plt.show()










