#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:55:17 2019

@author: acabbia
"""
from datetime import datetime
import os
import cobra
import pandas as pd
import grakel as gk
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, jaccard, squareform
from itertools import permutations

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity

from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree, TreeStyle
from ete3 import NCBITaxa

#%%
# loading models and data structures functions


def binary(model, ref_model):

    # init
    rxns = []
    mets = []
    genes = []

    for r in ref_model.reactions:
        if r in model.reactions:
            rxns.append(1)
        else:
            rxns.append(0)

    for m in ref_model.metabolites:
        if m in model.metabolites:
            mets.append(1)
        else:
            mets.append(0)

    for g in ref_model.genes:
        if g in model.genes:
            genes.append(1)
        else:
            genes.append(0)

    return rxns, mets, genes


def modelNet(model):
    #Returns a grakel.Graph object from a cobra.model object

    edges_in = []
    edges_out = []
    edges = []

    for r in model.reactions:
        # enumerate 'substrates -> reactions' edges
        substrates = [s.id for s in r.reactants]
        edges_in.extend([(s, r.id) for s in substrates])
        # enumerate 'reactions -> products' edges
        products = [p.id for p in r.products]
        edges_out.extend([(p, r.id) for p in products])

    # Join lists
    edges.extend(edges_in)
    edges.extend(edges_out)

    #labels
    label_m = {m.id: m.name for m in model.metabolites}
    label_r = {r.id: r.name for r in model.reactions}
    label_nodes = label_m
    label_nodes.update(label_r)
    label_edges = {p: p for p in edges}

    g = gk.Graph(edges, node_labels=label_nodes, edge_labels=label_edges)

    return g


def FBA(model, ref_model):

    ###### set obj and (minimal) bounds
    model.objective = model.reactions.get_by_id(
        [r.id for r in model.reactions if 'biomass' in r.id][0])

    #open all exchanges
    for e in model.reactions:
        e.bounds = -1000, 1000

    # optimize (normal FBA)
    sol = model.optimize()
    # flux distributions are appended following the index
    return sol.fluxes


def load_library(path, ref_model_path):
    '''
    loads models from library folder and prepares data structures for further analysis
    returns:

        - Binary matrices (rxn,met,genes) --> EDA and Jaccard
        - Graphlist --> Graph Kernels
        - Flux vectors matrix --> cosine similarity

    '''

    ref_model = cobra.io.read_sbml_model(ref_model_path)

    # Init
    reactions_matrix = pd.DataFrame(index=[r.id for r in ref_model.reactions])
    metabolite_matrix = pd.DataFrame(
        index=[m.id for m in ref_model.metabolites])
    gene_matrix = pd.DataFrame(index=[g.id for g in ref_model.genes])
    sol_df = pd.DataFrame(index=[r.id for r in ref_model.reactions])
    graphlist = []

    for filename in sorted(os.listdir(path)):
        model = cobra.io.read_sbml_model(path+filename)
        label = str(filename).split('.')[0]

        print("loading:", label)

        # 1: make binary matrices
        rxns, mets, genes = binary(model, ref_model)
        reactions_matrix[label] = rxns
        metabolite_matrix[label] = mets
        gene_matrix[label] = genes

        # 2: make graphlist
        graphlist.append(modelNet(model))

        # 3: make flux matrix
        fluxes = FBA(model, ref_model)
        sol_df[label] = fluxes

    return reactions_matrix, metabolite_matrix, gene_matrix, graphlist, sol_df

# Exploratory analysis functions


def boxplots(df, label):
    # Reactions/metabolites/genes content of the models, grouped by label

    groups = df.T.sum(axis=1).groupby(label)

    names = []
    data = []

    for g in groups:
        names.append(g[0])
        data.append(g[1].values)

    ax = sns.boxplot(data=data)
    ax.set_xticklabels(labels=names, rotation=90)

    #ax.get_figure().savefig(outfolder+'boxplots/'+c+'.png', dpi=1200, bbox_inches='tight')
    plt.show()

# distance matrix functions


def jaccard_DM(df):
    # returns square pairwise (jaccard) distance matrix between elements of df

    DM = pd.DataFrame(squareform(pdist(df.T, metric=jaccard)),
                      index=df.columns, columns=df.columns)

    return DM


def gKernel_DM(graphList):
    # returns 1 - kernel similarity matrix (i.e. distance)
    gkernel = gk.WeisfeilerLehman(
        base_kernel=gk.VertexHistogram, normalize=True)
    K = pd.DataFrame(gkernel.fit_transform(graphList))
    return 1-K


def FBA_cosine_DM(df):
    # returns cosine similarity between flux vectors

    #Remove Nan's
    sol_df = df.replace(np.nan, 0)
    processed = sol_df.T

    DM = pd.DataFrame(cosine_similarity(processed),
                      index=processed.index, columns=processed.index)
    return 1 - DM

# Clustering functions


def CalculateAccuracy(y, y_hat):

    accuracy = 0
    bestP = []
    perm = permutations(np.unique(y))

    for p in perm:

        tr = dict(zip(p, list(range(len(np.unique(y))))))
        y_tr = np.array([tr[v] for v in y])

        testAccuracy = accuracy_score(y_tr, y_hat)

        if testAccuracy > accuracy:
            accuracy = testAccuracy
            bestP.append((p, testAccuracy))

    P_df = pd.DataFrame(bestP)
    bestLabel = list(P_df.max()[0])

    inv_tr = dict(zip(list(range(len(np.unique(y)))), bestLabel))
    inv_y_hat = np.array([inv_tr[v] for v in y_hat])

    cm = confusion_matrix(y, inv_y_hat, bestLabel)

    return accuracy, bestLabel, cm


def HCClust(DM, trueLabel):

    HC = AgglomerativeClustering(n_clusters=len(
        pd.Series(trueLabel).unique()), affinity='precomputed', linkage='average').fit(DM)
    y_pred = HC.labels_

    accHC, bestLabHC, cmHC = CalculateAccuracy(trueLabel, y_pred)

    return accHC, bestLabHC, cmHC


def SCClust(DM, trueLabel):

    SC = SpectralClustering(n_clusters=len(
        pd.Series(trueLabel).unique()), affinity='precomputed').fit(1-DM)
    y_pred = SC.labels_

    accSC, bestLabSC, cmSC = CalculateAccuracy(trueLabel, y_pred)

    return accSC, bestLabSC, cmSC

# classification functions


def classify(DM, truelabel):
    # performs (10-fold CV) classification (with SVM and KNN), prints accuracy of retrieval of original labels

    # arguments:
    # DM = Distance matrix
    # label = (list) class label for each model

    # K_NN 10-Fold CV
    neigh = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    scores_K_NN = cross_val_score(
        neigh, DM.values, truelabel, cv=10, scoring='accuracy')
    print("Accuracy K-NN:", str(scores_K_NN.mean().round(2)),
          'CV error:', scores_K_NN.std().round(2))

    # Kernel SVM 10-fold CV
    K = 1-DM
    clf = SVC(kernel='precomputed', C=1)
    scores_K_SVM = cross_val_score(
        clf, K.values, truelabel, cv=10, scoring='accuracy')
    print("Accuracy SVM:", str(scores_K_SVM.mean().round(2)),
          'CV error:', scores_K_NN.std().round(2))

    return scores_K_NN, scores_K_SVM

# PhyloTree functions


def make_tree(DM):

    njtree = nj(DM, result_constructor=str)
    tree = Tree(njtree)
    return tree


def plot_tree(tree,  save=False, path=''):

    # style
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.mode = "c"
    ts.arc_start = -180
    ts.arc_span = 360

    #plot tree
    if save:
        tree.render(file_name=path, tree_style=ts)

    tree.show(tree_style=ts)


#%%
#####################################################################################################
###PATHS

path_PDGSMM = '/home/acabbia/Documents/Muscle_Model/models/merged_100/'
path_AGORA = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'

path_ref_PDGSMM = '/home/acabbia/Documents/Muscle_Model/models/HMR2.xml'
path_ref_AGORA = '/home/acabbia/Documents/Muscle_Model/models/AGORA_universe.xml'


### Labels
label_PDGSM = [s.split('_')[0] for s in sorted(os.listdir(path_PDGSMM))]

AGORA_taxonomy = pd.read_csv('/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv',
                             sep='\t').sort_values(by='organism')

# Replace Nan's with 'Other'
AGORA_taxonomy.replace(np.nan, 'Other', inplace=True)

# Replaces and aggregates classes with less than 10 samples into a new "Other" class
# to reduce the number of classes and computational time of the clustering

for c in ['phylum', 'oxygenstat', 'gram', 'mtype', 'metabolism']:
    for s in list(AGORA_taxonomy[c].value_counts()[AGORA_taxonomy[c].value_counts() < 10].index):
        AGORA_taxonomy[c].replace(s, 'Other', inplace=True)

# Merge 'Nanaerobe' with "Microaerophile" in oxygenstat variable
AGORA_taxonomy['oxygenstat'].replace(
    'Nanaerobe', 'Microaerophile', inplace=True)


# Make labels
label_AGORA_phylum = list(AGORA_taxonomy['phylum'].values)
label_AGORA_oxy = list(AGORA_taxonomy['oxygenstat'].values)
label_AGORA_gram = list(AGORA_taxonomy['gram'].values)
label_AGORA_type = list(AGORA_taxonomy['mtype'].values)

#%%
### Load GSMMs Libraries

rxns_PDGSM, met_PDGSM, gene_PDGSM, graphlist_PDGSM, flux_PDGSM = load_library(
    path_PDGSMM, path_ref_PDGSMM)
rxns_AGORA, met_AGORA, gene_AGORA, graphlist_AGORA, flux_AGORA = load_library(
    path_AGORA, path_ref_AGORA)

#%%
# Explorative Data Analysis (boxplots)
# Reactions/metabolites/(genes) content of the models, grouped by label

boxplots(rxns_AGORA, label_AGORA_gram)
boxplots(rxns_AGORA, label_AGORA_oxy)
boxplots(rxns_AGORA, label_AGORA_phylum)
boxplots(rxns_AGORA, label_AGORA_type)
boxplots(rxns_PDGSM, label_PDGSM)

boxplots(met_AGORA, label_AGORA_gram)
boxplots(met_AGORA, label_AGORA_oxy)
boxplots(met_AGORA, label_AGORA_phylum)
boxplots(met_AGORA, label_AGORA_type)
boxplots(met_PDGSM, label_PDGSM)

boxplots(gene_AGORA, label_AGORA_gram)
boxplots(gene_AGORA, label_AGORA_oxy)
boxplots(gene_AGORA, label_AGORA_phylum)
boxplots(gene_AGORA, label_AGORA_type)
boxplots(gene_PDGSM, label_PDGSM)

#%%

# make jaccard distance matrices
JD_PDGSM = jaccard_DM(rxns_PDGSM)
JD_AGORA = jaccard_DM(rxns_AGORA)

# make kernel distance matrices
GK_PDGSM = gKernel_DM(graphlist_PDGSM)
GK_AGORA = gKernel_DM(graphlist_AGORA)

# make FBA cosine distance matrices
COS_PDGSM = FBA_cosine_DM(flux_PDGSM)
COS_AGORA = FBA_cosine_DM(flux_AGORA)

#%%
##### Clustering
### HC
# JD

JD_HC_PDGSM_acc, JD_HC_PDGSM_pred_label, JD_HC_PDGSM_cm = HCClust(
    JD_PDGSM, label_PDGSM)

JD_HC_AGORA_acc_gram, JD_HC_AGORA_pred_label_gram, JD_HC_AGORA_cm_gram = HCClust(
    JD_AGORA, label_AGORA_gram)
JD_HC_AGORA_acc_oxy, JD_HC_AGORA_pred_label_oxy, JD_HC_AGORA_cm_oxy = HCClust(
    JD_AGORA, label_AGORA_oxy)
JD_HC_AGORA_acc_phylum, JD_HC_AGORA_pred_label_phylum, JD_HC_AGORA_cm_phylum = HCClust(
    JD_AGORA, label_AGORA_phylum)
JD_HC_AGORA_acc_type, JD_HC_AGORA_pred_label_type, JD_HC_AGORA_cm_type = HCClust(
    JD_AGORA, label_AGORA_type)

#GK
GK_HC_PDGSM_acc, GK_HC_PDGSM_pred_label, GK_HC_PDGSM_cm = HCClust(
    GK_PDGSM, label_PDGSM)

GK_HC_AGORA_acc_gram, GK_HC_AGORA_pred_label_gram, GK_HC_AGORA_cm_gram = HCClust(
    GK_AGORA, label_AGORA_gram)
GK_HC_AGORA_acc_oxy, GK_HC_AGORA_pred_label_oxy, GK_HC_AGORA_cm_oxy = HCClust(
    GK_AGORA, label_AGORA_oxy)
GK_HC_AGORA_acc_phylum, GK_HC_AGORA_pred_label_phylum, GK_HC_AGORA_cm_phylum = HCClust(
    GK_AGORA, label_AGORA_phylum)
GK_HC_AGORA_acc_type, GK_HC_AGORA_pred_label_type, GK_HC_AGORA_cm_type = HCClust(
    GK_AGORA, label_AGORA_type)

#COS
COS_HC_PDGSM_acc, COS_HC_PDGSM_pred_label, COS_HC_PDGSM_cm = HCClust(
    COS_PDGSM, label_PDGSM)

COS_HC_AGORA_acc_gram, COS_HC_AGORA_pred_label_gram, COS_HC_AGORA_cm_gram = HCClust(
    COS_AGORA, label_AGORA_gram)
COS_HC_AGORA_acc_oxy, COS_HC_AGORA_pred_label_oxy, COS_HC_AGORA_cm_oxy = HCClust(
    COS_AGORA, label_AGORA_oxy)
COS_HC_AGORA_acc_phylum, COS_HC_AGORA_pred_label_phylum, COS_HC_AGORA_cm_phylum = HCClust(
    COS_AGORA, label_AGORA_phylum)
COS_HC_AGORA_acc_type, COS_HC_AGORA_pred_label_type, COS_HC_AGORA_cm_type = HCClust(
    COS_AGORA, label_AGORA_type)

# Collect results

JD_HC_ACC_list = [JD_HC_PDGSM_acc, JD_HC_AGORA_acc_gram, JD_HC_AGORA_acc_oxy,
                  JD_HC_AGORA_acc_phylum, JD_HC_AGORA_acc_type]

GK_HC_ACC_list = [GK_HC_PDGSM_acc, GK_HC_AGORA_acc_gram, GK_HC_AGORA_acc_oxy,
                  GK_HC_AGORA_acc_phylum, GK_HC_AGORA_acc_type]

COS_HC_ACC_list = [COS_HC_PDGSM_acc, COS_HC_AGORA_acc_gram, COS_HC_AGORA_acc_oxy,
                   COS_HC_AGORA_acc_phylum, COS_HC_AGORA_acc_type]

### SC
# JD
JD_SC_PDGSM_acc, JD_SC_PDGSM_pred_label, JD_SC_PDGSM_cm = SCClust(
    JD_PDGSM, label_PDGSM)

JD_SC_AGORA_acc_gram, JD_SC_AGORA_pred_label_gram, JD_SC_AGORA_cm_gram = SCClust(
    JD_AGORA, label_AGORA_gram)
JD_SC_AGORA_acc_oxy, JD_SC_AGORA_pred_label_oxy, JD_SC_AGORA_cm_oxy = SCClust(
    JD_AGORA, label_AGORA_oxy)
JD_SC_AGORA_acc_phylum, JD_SC_AGORA_pred_label_phylum, JD_SC_AGORA_cm_phylum = SCClust(
    JD_AGORA, label_AGORA_phylum)
JD_SC_AGORA_acc_type, JD_SC_AGORA_pred_label_type, JD_SC_AGORA_cm_type = SCClust(
    JD_AGORA, label_AGORA_type)

#GK
GK_SC_PDGSM_acc, GK_SC_PDGSM_pred_label, GK_SC_PDGSM_cm = SCClust(
    GK_PDGSM, label_PDGSM)

GK_SC_AGORA_acc_gram, GK_SC_AGORA_pred_label_gram, GK_SC_AGORA_cm_gram = SCClust(
    GK_AGORA, label_AGORA_gram)
GK_SC_AGORA_acc_oxy, GK_SC_AGORA_pred_label_oxy, GK_SC_AGORA_cm_oxy = SCClust(
    GK_AGORA, label_AGORA_oxy)
GK_SC_AGORA_acc_phylum, GK_SC_AGORA_pred_label_phylum, GK_SC_AGORA_cm_phylum = SCClust(
    GK_AGORA, label_AGORA_phylum)
GK_SC_AGORA_acc_type, GK_SC_AGORA_pred_label_type, GK_SC_AGORA_cm_type = SCClust(
    GK_AGORA, label_AGORA_type)

#COS
COS_SC_PDGSM_acc, COS_SC_PDGSM_pred_label, COS_SC_PDGSM_cm = SCClust(
    COS_PDGSM, label_PDGSM)

COS_SC_AGORA_acc_gram, COS_SC_AGORA_pred_label_gram, COS_SC_AGORA_cm_gram = SCClust(
    COS_AGORA, label_AGORA_gram)
COS_SC_AGORA_acc_oxy, COS_SC_AGORA_pred_label_oxy, COS_SC_AGORA_cm_oxy = SCClust(
    COS_AGORA, label_AGORA_oxy)
COS_SC_AGORA_acc_phylum, COS_SC_AGORA_pred_label_phylum, COS_SC_AGORA_cm_phylum = SCClust(
    COS_AGORA, label_AGORA_phylum)
COS_SC_AGORA_acc_type, COS_SC_AGORA_pred_label_type, COS_SC_AGORA_cm_type = SCClust(
    COS_AGORA, label_AGORA_type)

# Collect results

JD_SC_ACC_list = [JD_SC_PDGSM_acc, JD_SC_AGORA_acc_gram, JD_SC_AGORA_acc_oxy,
                  JD_SC_AGORA_acc_phylum, JD_SC_AGORA_acc_type]

GK_SC_ACC_list = [GK_SC_PDGSM_acc, GK_SC_AGORA_acc_gram, GK_SC_AGORA_acc_oxy,
                  GK_SC_AGORA_acc_phylum, GK_SC_AGORA_acc_type]

COS_SC_ACC_list = [COS_SC_PDGSM_acc, COS_SC_AGORA_acc_gram, COS_SC_AGORA_acc_oxy,
                   COS_SC_AGORA_acc_phylum, COS_SC_AGORA_acc_type]

#### PLOT CLUSTERING RESULTS
HC_Clust_results_df = pd.DataFrame()
HC_Clust_results_df['Reaction Similarity (Jaccard)'] = JD_HC_ACC_list
HC_Clust_results_df['Network Similarity (Graph Kernel)'] = GK_HC_ACC_list
HC_Clust_results_df['Flux vector similarity (Cosine)'] = COS_HC_ACC_list
HC_Clust_results_df.index=['PDGSM, AGORA-Gram, AGORA-Oxygen, AGORA-Phylum, AGORA-Type']
HC_Clust_results_df.plot.bar(title='Hierarchical clustering: accuracy')


SC_Clust_results_df = pd.DataFrame()
SC_Clust_results_df['Reaction Similarity (Jaccard)'] = JD_SC_ACC_list
SC_Clust_results_df['Network Similarity (Graph Kernel)'] = GK_SC_ACC_list
SC_Clust_results_df['Flux vector similarity (Cosine)'] = COS_SC_ACC_list
SC_Clust_results_df.index=['PDGSM, AGORA-Gram, AGORA-Oxygen, AGORA-Phylum, AGORA-Type']
SC_Clust_results_df.plot.bar(title='Spectral clustering: accuracy')


#%%
#### classification

JD_PDGSM_knn, JD_PDGSM_svm = classify(JD_PDGSM, label_PDGSM)
JD_AGORA_knn_gram, JD_AGORA_svm_gram = classify(JD_AGORA, label_AGORA_gram)
JD_AGORA_knn_oxy, JD_AGORA_svm_oxy = classify(JD_AGORA, label_AGORA_oxy)
JD_AGORA_knn_phylum, JD_AGORA_svm_phylum = classify(
    JD_AGORA, label_AGORA_phylum)
JD_AGORA_knn_type, JD_AGORA_svm_type = classify(JD_AGORA, label_AGORA_type)

GK_PDGSM_knn, GK_PDGSM_svm = classify(GK_PDGSM, label_PDGSM)
GK_AGORA_knn_gram, GK_AGORA_svm_gram = classify(GK_AGORA, label_AGORA_gram)
GK_AGORA_knn_oxy, GK_AGORA_svm_oxy = classify(GK_AGORA, label_AGORA_oxy)
GK_AGORA_knn_phylum, GK_AGORA_svm_phylum = classify(
    GK_AGORA, label_AGORA_phylum)
GK_AGORA_knn_type, GK_AGORA_svm_type = classify(GK_AGORA, label_AGORA_type)

COS_PDGSM_knn, COS_PDGSM_svm = classify(COS_PDGSM, label_PDGSM)
COS_AGORA_knn_gram, COS_AGORA_svm_gram = classify(COS_AGORA, label_AGORA_gram)
COS_AGORA_knn_oxy, COS_AGORA_svm_oxy = classify(COS_AGORA, label_AGORA_oxy)
COS_AGORA_knn_phylum, COS_AGORA_svm_phylum = classify(
    COS_AGORA, label_AGORA_phylum)
COS_AGORA_knn_type, COS_AGORA_svm_type = classify(COS_AGORA, label_AGORA_type)

#%%

##### Trees comparison
# make ref tree (NCBI)

ncbi = NCBITaxa()
ncbi.update_taxonomy_database()

NCBI_ID = list(AGORA_taxonomy['ncbiid'].dropna().values)
NCBI_tree = ncbi.get_topology(NCBI_ID)

# Ugly way to convert "phyloTree" obj into "Tree" obj for comparison with other trees
NCBI_tree.write(
    format=1, outfile="/home/acabbia/Documents/Muscle_Model/GSMM-distance/trees/NCBI_tree.nw")
NCBI_tree = Tree(
    "/home/acabbia/Documents/Muscle_Model/GSMM-distance/trees/NCBI_tree.nw", format=1)

# fix non zero diagonal values in FBA DM
COS_AGORA[COS_AGORA < 10e-10] = 0
COS_PDGSM[COS_PDGSM < 10e-10] = 0

# Convert Distance matrices into skbio.Distance_matrix objects

JD_AGORA_DM = DistanceMatrix(JD_AGORA)
GK_AGORA_DM = DistanceMatrix(GK_AGORA)
COS_AGORA_DM = DistanceMatrix(COS_AGORA)

JD_PDGSM_DM = DistanceMatrix(JD_PDGSM)
GK_PDGSM_DM = DistanceMatrix(GK_PDGSM)
COS_PDGSM_DM = DistanceMatrix(COS_PDGSM)

## make trees
TREE_JD_AGORA = make_tree(JD_AGORA_DM)
TREE_GK_AGORA = make_tree(GK_AGORA_DM)
TREE_COS_AGORA = make_tree(COS_AGORA_DM)

TREE_JD_PDGSM = make_tree(JD_PDGSM_DM)
TREE_GK_PDGSM = make_tree(GK_PDGSM_DM)
TREE_COS_PDGSM = make_tree(COS_PDGSM_DM)

#### What to do with PDGSM Tree?


#%%
# dictionary to translate between model_taxonomy.index (GK and JD trees) and NCBI_id (NCBI tree)

idx_str = [str(i) for i in list(AGORA_taxonomy.index)]
NCBI_str = [str(i) for i in NCBI_ID]

translator = dict(zip(idx_str, NCBI_str))

#Annotate GK and JD trees with NCBI id's
for tree in [TREE_JD_AGORA, TREE_GK_AGORA, TREE_COS_AGORA]:
    for leaf in tree:
        leaf.name = translator[leaf.name]

##### comparisons with ref
REF_JD_AGORA = TREE_JD_AGORA.compare(NCBI_tree, unrooted=True)
REF_GK_AGORA = TREE_GK_AGORA.compare(NCBI_tree, unrooted=True)
RES_COS_AGORA = TREE_COS_AGORA.compare(NCBI_tree, unrooted=True)

### comparisons between metrics
JD_GK_AGORA = TREE_JD_AGORA.compare(TREE_GK_AGORA, unrooted=True)
JD_COS_AGORA = TREE_JD_AGORA.compare(TREE_COS_AGORA, unrooted=True)
GK_COS_AGORA = TREE_GK_AGORA.compare(TREE_COS_AGORA, unrooted=True)
