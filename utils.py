from __future__ import print_function
import time
import numpy as np
import scipy as sc
import scipy.sparse as sp
import torch
import os
import re
import networkx as nx
import torch.utils.data as Data
import torch.optim.lr_scheduler
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import sys
import csv
import gzip
import logging
import json
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils import shuffle

SEQ_MAX_LEN = 210 # the length of protein sequence


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

##### JSON modules #####
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4)
def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data
##### JSON modules #####

def load_ikey2smiles():
    file_path='data/Integrated/chemicals'
    os.path.exists(file_path)
    ikey2smiles={}
    with open(os.path.join(file_path,'integrated_chemicals.tsv'),'r') as fin:
        for line in fin:
            line=line.strip().split('\t')
            ikey=line[1]
            smi=line[2]
            ikey2smiles[ikey]=smi
    return ikey2smiles

def padding(batch, max_len, pad):
    padded = []
    lengths = []
    for seq in batch:
        seq = seq[:min(len(seq), max_len)]
        lengths.append(len(seq))
        seq = seq + [pad] * (max_len - len(seq))
        padded.append(seq)
    return padded, lengths
def get_lstm_embedding(batch_repr,max_len=SEQ_MAX_LEN,pad=0):
    batch_repr, lengths = padding(batch_repr, max_len, pad)
    with torch.no_grad():
        batch_repr = Variable(torch.LongTensor(batch_repr))
    logging.debug("utils.get_lstm_embedding: batch_repr {}".format(batch_repr.size()))
    return batch_repr
def load_mtl_edges_from_file(edgefile,allowed_uniprots=None,sep=',',header=True):
    #default data format:
    #InChIKey,UniProt,Binary,pKi,pKd,pIC50
    #MAEHEIXUINDDHE-UHFFFAOYSA-N,P48736,1,6.130182,6.130182,6.130182
    
    #missing entries (presented as 'nan') are converted to -1
    
    edges={};ikeys=[];uniprots=[]
    count_skipped=0
    count_loaded=0
    with open(edgefile,'r') as f:
        if header:
            next(f)
        for line in f:
            line=line.strip().split(sep)
            ikey=line[0]
            uni=line[1]
            if allowed_uniprots and (uni not in allowed_uniprots):
                count_skipped+=1
                continue
            ikeys.append(ikey)
            uniprots.append(uni)
            try:
                b=np.float(line[2])
            except:
                b=-1
            try:
                ki=np.float(line[3])
            except:
                ki=-1
            try:
                kd=np.float(line[4])
            except:
                kd=-1
            try:    
                ic=np.float(line[5])
            except:
                ic=-1
            b=-1 if np.isnan(b) else b
            ki=-1 if np.isnan(ki) else ki
            kd=-1 if np.isnan(kd) else kd
            ic=-1 if np.isnan(ic) else ic
            val=(b,ki,kd,ic)
            edge=ikey+'\t'+uni
            edges[edge]=val
            count_loaded+=1
    logging.info("{} edges loaded. {} edges (not-allowed-uniprots) skipped from {}".format(count_loaded,
                                                                                           count_skipped,
                                                                                           edgefile))
    ikeys=list(set(ikeys));uniprots=list(set(uniprots))
    return edges,ikeys,uniprots
def load_edges_from_file(edgefile,sep=',',header=True):
    #default data format:
    #InChIKey,UniProt,activity (sep=',')
    #MAEHEIXUINDDHE-UHFFFAOYSA-N,P48736,6.130
    
    edges={};ikeys=[];uniprots=[]
    count_skipped=0
    count_loaded=0
    with open(edgefile,'r') as f:
        if header:
            next(f)
        for line in f:
            line=line.strip().split(sep)
            ikey=line[0]
            uni=line[1]
            # if allowed_uniprots and (uni not in allowed_uniprots):
            #     count_skipped+=1
            #     continue
            ikeys.append(ikey)
            uniprots.append(uni)
            val=float(line[2])
            edge=ikey+'\t'+uni
            edges[edge]=val
            count_loaded+=1
    logging.info("{} edges loaded. {} edges (not-allowed-uniprots) skipped from {}".format(count_loaded,
                                                                                           count_skipped,
                                                                                           edgefile))
    ikeys=list(set(ikeys));uniprots=list(set(uniprots))
    return edges,ikeys,uniprots

def load_dict(path):
    """ Load a dictionary and a corresponding reverse dictionary from the given file
    where line number (0-indexed) is key and line string is value. """
    retdict = list()
    rev_retdict = dict()
    with open(path) as fin:
        for idx, line in enumerate(fin):
            text = line.strip()
            retdict.append(text)
            rev_retdict[text] = idx
    return retdict, rev_retdict

def load_repr(path, config, node_list):
    """ Load the representations of each node in the `node_list` given
    the representation type and configurations.

    Args:
        path: Path of the graph data directory
        config: Node configuration JSON object
        node_list: The list of nodes for which to load representations

    Returns:
        repr_info: A dictionary that contains representation information
        node_list: List of nodes with loaded representations, the change
        is in-place though.
    """
    repr_type = config['representation']
    if repr_type == TYPE_MOLECULE:
        return load_molecule_repr(path, config, node_list)
    elif repr_type == TYPE_SEQUENCE_PSSM:
        return load_pssm_repr(path, config, node_list)
    else:
        raise ValueError("{0} Node type not supported!".format(repr_type))

def load_molecule_repr(path, config, node_list):
    import deepnet.fingerprint.features as fp_feature
    graph_vocab_path = os.path.join(path, config['graph_path'])
    graph_list, _ = load_dict(graph_vocab_path)
    for node, graph in zip(node_list, graph_list):
        node.set_data(graph)
    info = dict(embedding_type=TYPE_MOLECULE,
                atom_size=fp_feature.num_atom_features(),
                bond_size=fp_feature.num_bond_features())
    return info, node_list

def load_uniprot2pssm(max_len=512,padding=0):
    #maximum sequence length: max_len
    #pssm padded with zeros if len<max_len
    base_path='data/protein/'
    pssm_dir=base_path+'kinase_domain_pssm_uniref50/'
    #protfile=base_path+'prot_bsite_sample' #padding test
    protfile=base_path+'prot_bsite'
    uniprot2pssm={}
    pssm_files=os.listdir(pssm_dir)
    manual_dict={'P52333_JH1domain-catalytic':'P52333_Kin.Dom.2-C-terminal.dat',
	    'Q9P2K8_Kin.Dom.2,S808G':'Q9P2K8_S808G_Kin.Dom.2-C-terminal.dat',
	    'P23458' :'P23458_JH2domain-pseudokinase.dat',
	    'P29597' :'P29597_JH2domain-pseudokinase.dat',
	    'O75582' :'O75582_Kin.Dom.1-N-terminal.dat',
	    'Q15418' :'Q15418_Kin.Dom.1-N-terminal.dat',
	    'Q9P2K8' :'Q9P2K8_Kin.Dom.1-N-terminal.dat',
	    'Q9UK32' :'Q9UK32_Kin.Dom.2-C-terminal.dat'}
    with open(protfile,'r') as f:
        for line in f:
            uniprot=line.strip()
            line=line.strip()
            line=line.replace('(','_').replace(')','')
            line=line.replace('-nonphosphorylated','').replace('-phosphorylated','').replace('-autoinhibited','')

            matchkd=re.search(r'Kin\.Dom',line,re.I)
            matchjh=re.search(r'JH\ddomain',line,re.I)
            if line in list(manual_dict.keys()):
                fname=manual_dict[line]
            elif matchkd:
                matchkd=re.search(r'Kin\.Dom\.(\d)',line,re.I)
                if matchkd is None:
                    fname=line+'.dat'
                elif matchkd.group(1)==str(1):
                    fname=line+'-N-terminal.dat'
                elif matchkd.group(1)==str(2):
                    fname=line+'-C-terminal.dat'
            elif matchjh:
                fname=line+'.dat'
            else:
                fname=line+'.dat'
            if fname not in pssm_files:
                fname=line.replace('\.dat','')+'_Kin.Dom.dat'
                #print("PSSM file {} not found for protein {}".format(fname,line))
            
            pssm=[]
            with open(pssm_dir+fname,'r') as f:
                for line in f:
                    line=line.strip().lstrip().split()
                    if len(line)==0: #empty line
                        continue
                    else:
                        try:
                            resnum=int(line[0])
                        except: #non-pssm field
                            continue
                        res_vector=np.array(line[2:22],dtype=np.float32)
                        pssm.append(res_vector)
            pssm=np.array(pssm,dtype=np.float32)
            if pssm.shape[0] > max_len:
                print("Sequence length for {0} ({1}) is greater than {2}. Truncated to {2}".format(uniprot,pssm.shape[0],max_len))
                pssm=pssm[:max_len,:]
            else:
                pssm=np.pad(pssm,((0,max_len-pssm.shape[0]),(0,0)),'constant',constant_values=padding) #pad to the bottom
            uniprot2pssm[uniprot]=pssm
            #print("PSSM shape {} loaded for {} from file {}".format(uniprot2pssm[uniprot].shape,uniprot,fname))
    for gpcr_pssm_file in os.listdir(os.path.join(base_path,'gpcr_pssm_uniref50')):
        pssm=[]
        uniprot=gpcr_pssm_file.strip().split('.')[0]
        with open(os.path.join(os.path.join(base_path,'gpcr_pssm_uniref50'),gpcr_pssm_file),'r') as f:
            for line in f:
                line=line.strip().lstrip().split()
                if len(line)==0: #empty line
                    continue
                else:
                    try:
                        resnum=int(line[0])
                    except: #non-pssm field
                        continue
                    res_vector=np.array(line[2:22],dtype=np.float32)
                    pssm.append(res_vector)
        pssm=np.array(pssm,dtype=np.float32)
        if pssm.shape[0] > max_len:
            print("Sequence length for {0} ({1}) is greater than {2}. Truncated to {2}".format(uniprot,pssm.shape[0],max_len))
            pssm=pssm[:max_len,:]
        else:
            pssm=np.pad(pssm,((0,max_len-pssm.shape[0]),(0,0)),'constant',constant_values=padding) #pad to the bottom
        uniprot2pssm[uniprot]=pssm
    return uniprot2pssm

def load_uniprot2singletrepr(binding_site=False):
    #set binding_site=True to obtain representations for only binding sites
    #if binding_site=False, representations are for whole protein sequences
    base_path = 'data/protein/'
    #if binding_site: #temporarily commented
    #    repr_file = base_path + 'prot_bsite.repr'
    #    id_file = base_path + 'prot_bsite'
    #else:
    #    repr_file = base_path + 'prot.repr'
    #    id_file = base_path + 'prot'

    #for gpcr models
    repr_file = base_path + 'gpcr_prot.repr'
    id_file = base_path + 'gpcr_prot'
    idx2id = {}
    id2repr = {}
    with open(id_file,'r') as f:
        for idx,line in enumerate(f):
            if line == '':
                continue
            line=line.strip()
            idx2id[idx]=line
    with open(repr_file,'r') as f:
        for idx,line in enumerate(f):
            if line == '':
                continue
            id2repr[idx2id[idx]] = [int(res) for res in line.strip().split()]
    return id2repr

def get_mutant_triplets(genesymbol,mutations=None):
    #gene symbol: e.g.H2N3I5_PONAB, mutations: e.g.'T35A', or 'T35A,K56L,I141M'
    seqstart,seqend=seqdict[genesymbol]['position'].strip().split('-')
    aligned_seq=seqdict[genesymbol]['aligned_sequence']
    aligned_seq_residues=[res for res in aligned_seq]
    if mutations:
        print("mutations {}".format(mutations))
        for mut in mutations.strip().split(','):
            searchobj=re.search(r'([A-Za-z])([0-9]+)([A-Za-z])',mut,re.I|re.M)
            #print("processing mutation {}".format(mut))
            if searchobj is None:
                #print("{} skipped".format(mut))
                continue
            from_aa=searchobj.group(1).lower()
            position=int(searchobj.group(2))
            to_aa=searchobj.group(3).lower()
            if position>int(seqend) or position<int(seqstart):
                #given mutation out of range, cannot be applied in the triplets
                #print("mutation {} not within the aligned sequence range".format(mut))
                continue
            else:
                i=0 #nongap residue number - seqstart
                for j,aa in enumerate(aligned_seq_residues):
                    if aa in ['.','-']:
                        continue
                    resnum=int(seqstart)+i
                    if resnum==position:
                        if aa.lower()==from_aa:
                            aligned_seq_residues[j]=to_aa
                            #print("Mutation {} applied".format(mut))
                            break
                        else:
                            print("Residue mismtach for {}. {} found instead of {} at {}".format(
                                mut,aa.lower(),from_aa,position))
                    i+=1
    else:
        print("Wild type")
    aligned_seq_mut=''.join(aligned_seq_residues)                
    lgapsearch=re.search(r'^(\.+)[a-z]',aligned_seq_mut,re.I|re.M)
    tgapsearch=re.search(r'[a-z](\.+)$',aligned_seq_mut,re.I|re.M)
    try:
        lgapcount=len(lgapsearch.group(1))
    except:
        lgapcount=0
    try:
        tgapcount=len(tgapsearch.group(1))
    except:
        tgapcount=0
    seq=[]
    for i in selected_positions:
        if i in list(range(0,lgapcount)) or i in list(range(aligned_len-tgapcount,aligned_len)):
            #leading or trailing gap
            seq.append(padchar)
        else:
            seq.append(aligned_seq_mut[i].lower().replace('.',padchar).replace('-',gapchar))
    sentence=' '.join(seq)
    words=[]
    for i in range(len(seq)-2):
        words.append(''.join(seq[i:i+3]))
    return ' '.join(words)


if __name__=='__main__':
    #uniprot2pssm=load_uniprot2pssm()
    ikey2smiles=load_ikey2smiles()
    import pickle
    with open('ikey2smiles.pickle', 'wb') as handle:
        pickle.dump(ikey2smiles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('ikey2smiles.pickle','rb') as handle:
        ikey2smiles2 = pickle.load(handle)
    if ikey2smiles==ikey2smiles2:
        print("ikey2smiles successfully saved to {}".format('ikey2smiles.pickle'))
    else:
        print("Error occurred. saved pickle object and loaded object are not equal")
#    print(uniprot2pssm['O60674(JH1domain-catalytic)'])
#    print(uniprot2pssm['O60674(JH2domain-pseudokinase)'])


