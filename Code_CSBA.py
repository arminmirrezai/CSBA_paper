# -*- coding: utf-8 -*-
"""
Created on Mon Nov 1 17:24:15 2021

@author: Armin Mirrezai 
"""

# %% import packages


import pandas as pd
import numpy as np
import json
import re
from scipy.optimize import fsolve
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


# %% utility functions


def import_data(file):
    """
    import_data imports data and returns it in a convenient pandas dataframe
    
    :param file: JSON file
    :return: pandas dataframe, the data in pandas dataframe format
    """ 
    
    # opening JSON file
    f = open(file)
     
    # returns JSON object as a dictionary
    data = json.load(f)
     
    # closing file
    f.close()
    
    # create a dataframe
    df_list = []
    for model in data.values(): 
        
        # loop over model instances
        for instance in model:
            
            # add product instance to list
            df_list.append(instance)
    
    return pd.DataFrame(df_list)


def string_replace(text_data, orig_text, std_text):
    """
    string_replace replaces the original strings into the standardized strings
    
    :param text_data: string or pandas data fram column
    :orig_text: list of strings to be replaced 
    :std_text: list of strings to replace the original strings with 
    :return std_output, the standardized strings
    """
    
    # the num strings to be replaced should match the no of std replacement strings
    if len(orig_text) != len(std_text):
        return print("ERROR: length of inputs do not match")
    
    # initialize output
    std_text = text_data
    
    # loop over list of string that need to be replaced
    for i in range(len(orig_text)):
        
        # check if input data is df column or regular string
        if type(text_data) == pd.Series:
            std_text = std_text.apply(lambda col: col.replace(orig_text[i], std_text[i]))
        else:
            std_text = std_text.replace(orig_text[i], std_text[i])

    return std_text


def data_cleaning(df):
    """
    data_cleaning preprocesses the webshop data so that different data entries 
    with the same menaing have a standardized format
    
    :param df: pandas dataframe
    :return: pandas dataframe
    """ 
    
    # change all titles to lowercase
    df['title'] = df['title'].apply(lambda col: col.lower())
    
    # standardize all titles
    orig_text = [" -hz", "-hz", " hertz", "hertz", " hz",
                 " inches", "inches", ' "', '"', ' ”', '”', " -inch", "-inch", " inch",
                 "!", "[", "]", "(", ")", "/", "-", "_", "."]
    std_text = 5 * ['hz'] + 9 * ['inch'] + 9 * [""]
    
    # iterate over all titles
    i = 0
    for title in df['title']:
        title = string_replace(title , orig_text, std_text)
        df['title'][i] = title
        i += 1
    
    return None

    
def model_word_extraction(df):
    """
    model_word_extraction extracts all the model words from the titles 
    and appends brand name and product type 
    
    :param df: pandas dataframe
    :return: list of distinct model words obtained from the titles, 
             list of product titles and a corresponding list of modelIDs
    """ 
    
    # obtain titles and modelIDs corresponding to titles
    titles = list(df['title'])
    modelIDs = list(df['modelID'])
    
    # regex for model words from title
    re_title_mws = "([azA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"

    # list of model words from title
    title_mws = [] 

    # iterate over all titles
    for title in titles:
        
        # extract models words from titles using regex
        mws_regex = re.findall(re_title_mws, title)
        title_mws += [mws_regex[i][0] for i in range(len(mws_regex))]

    # add brands to model words
    brands = ["lg","philips","samsung","toshiba","nec","supersonic","sharp","viewsonic", "sony", "coby","sansui",
              "tcl","hisense","optoma","haier","panasonic","rca","naxa","vizio","magnavox", "sanyo", "insignia",
              'jvc','proscan','venturer','westinghouse','sceptre']     
    mws = title_mws + brands
    
    # remove duplicates
    mws = list(set(mws))
    
    return titles, modelIDs, mws

def binary_representation(titles, mws):
    """
    binary_representation create a matrix with the binary vector representation with the 
    model words on the rows and the product on the columns
    
    :param titles: list of product titles
    :param mws: list of model words
    :return: numpy matrix with binary vector representation with the model words
    on the rows and the products on the columns
    """         
    # create binary columns
    col_k = 0
    
    # initialize binary matrix representation
    binary_rep = np.zeros((len(mws), len(titles)))
    
    # loop over all titles
    for title in titles:
        title_words = title.split()
        
        # loop over all model words
        for i in range(len(mws)):
            
            # add 1's if model word occurs in title
            for word in title_words:
                if mws[i] == word:
                    binary_rep[i, col_k] = 1
                    
        # move to next column
        col_k = col_k + 1
        
    return binary_rep 

def minhash_array(mws, num_permutations):
    """
    minhash_array creates an array of permutations that can be used to minhash
    
    :param mws: list of model words
    :param num_permutations: the desired number of permutations, i.e. the 
                                dimension of the minhash_array
    :return: minhash array
    """   
    # initialize the length of the permutations
    num_mws = len(mws)
    
    # initialize array
    minhash_arr = np.zeros((num_mws, num_permutations))
    
    # create permatuations
    for i in range(num_permutations):
        permutation = np.random.permutation(num_mws) + 1
        minhash_arr[:, i] = permutation.copy()
        
    return minhash_arr

def create_signature(minhash_arr, product):
    """
    create_signiature create a signature for a binary representation of a 
    product
    
    :param minhash_arr: minhash array
    :param product: the binary representation of a product
    :return: signature for bin_rep of a product
    """   
    # obtain index of 1's in the binary representation of the product
    idx = np.nonzero(product)[0].tolist()
    
    # obtain permutation values at index locations 1's
    perm_values = minhash_arr[idx, :]
    
    # the number of the first row that has a 1
    signature = np.min(perm_values, axis=0)
    
    return signature

def solve_r_b(params, *data):
    """
    solve_r_b is a function that entails two equations, wich can be solved
    to find the optimal value of r and  using n and t
    
    :param params: list with r (current estimate of rows) and 
                    b (current estimate of bands)
    :param data: values of n (length of signature) and t (predefined threshold)
    :return: list with (i) difference between n and r*b and 
            (ii) difference between t and (1/b)^(1/r)
    """   
    t, n = data
    r = params[0]
    b = params[1]
    return [n - r*b, t - (1/b)**(1/r)]

def shingles(k, string):
    """
    shingles converts a string into a k-shingle
    
    :param k: length of substrings 
    :param string: string that is being shingled
    :return: k-shingle of a string
    """   
    # initialize dictionary for shingles
    shingles = dict()
    
    # loop over string
    for i in range(len(string) - k + 1):
        
        # create shingle
        shingle = string[i:i + k]
        
        # check the num of occurances of shingle
        count = shingles.get(shingle)
        
        # update the num of occurances of shingle
        if count:
            shingles[str(shingle)] = int(count + 1)
        else:
            shingles[str(shingle)] = 1
            
    return shingles

def q_gram(str_a, str_b):
    """
    q_gram calculates the q-gram distance between two strings
    
    :param str_a: first string to be compared
    :param str_b: second string to be compared
    :return: q-gram distance of two string
    """ 
    # convert strings in to shingles
    shingles_a = shingles(3, str_a)
    shingles_b = shingles(3, str_b)
    
    # create union of all shingles in both strings
    union = set()
    for shingle in shingles_a.keys():
        union.add(shingle)
    for shingle in shingles_b.keys():
        union.add(shingle)
     
    # initialize distance
    distance = 0
    for shingle in union:
        
        # compare occurance of shingle in string a and b
        count_a, count_b = 0, 0
        if shingles_a.get(shingle) is not None:
            count_a = int(shingles_a.get(shingle))
        if shingles_b.get(shingle) is not None:
            count_b = int(shingles_b.get(shingle))
        
        # apply manhatten distance
        distance += abs(count_a - count_b)
        
    return distance

def true_duplicates(modelIDs):
    """
    true_duplicates computes the true duplicate pairs in the data
    
    :param modelIDs: a list with the modelIDs of all products
    :return: true_pairs is a list with the true duplicate pairs
    """ 
    # create a dictionary with the true duplicates
    true_duplicates = defaultdict(set)
    for i in range(len(modelIDs)):
        true_duplicates[modelIDs[i]].add(i)
        
    # initialize set with final duplicate pairs
    true_pairs = set()
    
    # loop over duplicates
    for duplicates in true_duplicates.values():
        
        # check if multiple products are in the same pair
        if len(duplicates) > 1:
            
            # compute final true pairs from true duplicates
            for pair in combinations(duplicates, 2):
                true_pairs.add(pair)
        
    return true_pairs

def F1_measure(found_pairs, true_pairs):
    """
    F1_measure calcutes the F1_measure
    
    :param found_pairs: the list of found pairs by EC
    :param true_pairs: the list of true duplicate pairs
    :return: 
    """ 
    TP = 0 
    FP = 0
    
    for found_pair in found_pairs:
        count = 0 
        for true_pair in true_pairs:
            if tuple(found_pair) == tuple(true_pair):
                count += 1
        
        if count > 0:
            TP += 1
        else:
            FP += 1
    
    # FN = len(true_pairs) - TP
    
    PC = TP / len(true_pairs)
    PQ = TP / (FP + TP)
    
    # F1 = TP / (TP + 0.5 * (FP + FN))
    F1 = (2 * PC * PQ ) / (PC + PQ)
    
    return F1, PQ, PC

# %% LSH and EC algorithm


def LSH(signatures, t):
    """
    LSH performs the locality sensitive hashing. It basically hashes columns 
    to many buckets, and makes elements of the same bucket candidate pairs
    
    :param signatures: 
    :param t: predefined threshold value
    :return: set of candidate pairs 
    """   
    # obtain signature matrix dimensions
    p, n = signatures.shape
    
    # solve r and b for the given t and n
    data = (t, n)
    params = fsolve(solve_r_b, [5,10], args=data, maxfev=10000)
    b = int(np.floor(params[1]))
    
    # initialize hash buckets
    hash_buckets = defaultdict(set)
    
    # split signatures in bands according to b
    bands = np.array_split(signatures, b, axis=1)
    
    # loop over bands
    for i, band in enumerate(bands):
        
        # loop over products
        for j in range(p):
            
            # create key to hash product to one of the hash buckets
            # made it a tuple to get hashable data type
            # added str(i) to avoid key collisions with other bands
            band_id = tuple([str(i)] + list(band[j,:]))
            hash_buckets[band_id].add(j)
    
    # initialize set of candidate pairs
    candidate_pairs = set()
    
    # loop over hash buckets
    for bucket in hash_buckets.values():
        
        # check if multiple products are hashed to the same bucket
        if len(bucket) > 1:
            
            # add candidate pairs
            for pair in combinations(bucket, 2):
                candidate_pairs.add(pair)
                
    return candidate_pairs

def dissimilarity_matrix(candidate_pairs, signatures, titles):
    """
    dissimilarity_matrix computes the dissimilarity matrix for products
    
    :param candidate_pairs: set of candidate pairs
    :param signatures: signature matrix
    :param titles: list of all product titles
    :return: dissimilarity matrix of all products
    """ 
    # initialize dissimilarity matrix
    dissim_matrix = np.ones((len(signatures[:,0]), len(signatures[:,0]))) * 999999
    
    # loop over all candidate pairs
    for pair in candidate_pairs:  
        
        # obtaining the indices of the potential pairs
        prod_1, prod_2 = pair

        # initialize the similarity for each pair at zero
        similarity = 0
        
        # calculate the similarity between the product titles
        title_1 = [x for x in titles[prod_1].split()]
        title_2 = [y for y in titles[prod_2].split()]
        num_dup_words = len(set(title_1) & set(title_2))
        title_sim = num_dup_words / len(set(title_1 + title_2))
            
        # calculate the similarity of the titles with qgram distance
        title_1_str = "".join(title_1)
        title_2_str = "".join(title_2)
        qgram_sim = q_gram(title_1_str, title_2_str)
        
        # calculate total similarity 
        similarity = title_sim + qgram_sim
        
        # transform similarity to dissimilarity 
        dissim_matrix[prod_1, prod_2] = 1 - similarity   
        
    return dissim_matrix

def EC(dissim_matrix):
    """
    EC computes the final pairs by clustering the candidate pairs found by LSH
    
    :param dissim_matrix: dissimilarity matrix of all products
    :return: clst_candidate_pairs are the clusters found by MSM and 
            final_pairs are the final pairs found by MSM
    """ 
    # cluster the dissimilarity matrix
    epsilon = 0
    clst = AgglomerativeClustering(n_clusters=None, affinity='precomputed', 
                                       linkage='single', distance_threshold=epsilon)
    
    # obtain cluster labels
    clusters = clst.fit_predict(dissim_matrix)

    # create a dictionary with the clusters
    clst_candidate_pairs = defaultdict(set)
    for i in range(len(clusters)):
        clst_candidate_pairs[clusters[i]].add(i)
        
    # initialize set with final pairs
    final_pairs = set()
    
    # loop over clusters
    for cluster in clst_candidate_pairs.values():
        
        # check if multiple products are in the same cluster
        if len(cluster) > 1:
            
            # compute final pairs from clusters
            for pair in combinations(cluster, 2):
                final_pairs.add(pair)

    return final_pairs
            

 #%% run script

# working directory
# cd documents/School/CS for Business Analytics 21-22/paper/code/csba_paper

# import data
df = import_data('TVs-all-merged.json')
    
# clean titles
data_cleaning(df)

# create a list of thresholds
t_list = np.arange(0,1,0.1)

# number of bootstraps
bootstrap = 10

# create list of measurements
FC_total_LSH = []
PQ_total_LSH = []
PC_total_LSH = []
F1_total_LSH = []

FC_total_EC = []
PQ_total_EC = []
PC_total_EC = []
F1_total_EC = []

# loop over all possible thresholds
for i in range(len(t_list)):
    
    # create list of measurement for each t
    FC_thres_LSH = []
    PQ_thres_LSH = []
    PC_thres_LSH = []
    F1_thres_LSH = []
    
    FC_thres_EC = []
    PQ_thres_EC = []
    PC_thres_EC = []
    F1_thres_EC = []


    # create 5 bootstraps
    for j in range(bootstrap):
        
        # generate training data
        bootstrap_training = df.sample(frac=0.63, replace=True, random_state=j)
        
        # generate test data
        bootstrap_test = df[~df.index.isin(list(bootstrap_training.index))]
        
        # extract the titles, modelIDs and model words
        titles, modelIDs, mws = model_word_extraction(bootstrap_training) 
        
        # number of products
        n = len(modelIDs)
        
        # get binary representation of titles
        binary_rep = binary_representation(titles, mws)
        
        # create minhash array
        num_minhash_func = int(np.floor(0.5*n))
        minhash_arr = minhash_array(mws, num_minhash_func)
        
        # create signatures
        signatures = []
        for product in range(n):
            signatures.append(create_signature(minhash_arr, binary_rep[:,product]))
        signatures = np.stack(signatures)
        
        # perform lsh
        t = t_list[i]
        candidate_pairs = LSH(signatures, t)
        
        # perform clustering
        dissim_matrix = dissimilarity_matrix(candidate_pairs, signatures, titles)
        found_pairs = EC(dissim_matrix)
        
        # compute f1-measure, pair quality and pair completeness
        true_pairs = true_duplicates(modelIDs)
        F1_LSH, PQ_LSH, PC_LSH = F1_measure(candidate_pairs, true_pairs)
        F1_EC, PQ_EC, PC_EC = F1_measure(found_pairs, true_pairs)
        
        # compute fraction of comparisons
        FC_LSH = len(candidate_pairs) / (n*(n-1)/2)
        FC_EC = len(found_pairs) / (n*(n-1)/2)
        
        # save measurement for bootstrap j
        FC_thres_LSH.append(FC_LSH)
        PQ_thres_LSH.append(PQ_LSH)
        PC_thres_LSH.append(PC_LSH)
        F1_thres_LSH.append(F1_LSH)
        
        FC_thres_EC.append(FC_EC)
        PQ_thres_EC.append(PQ_EC)
        PC_thres_EC.append(PC_EC)
        F1_thres_EC.append(F1_EC)
    
    # save measurement for threshold t
    FC_total_LSH.append(np.mean(FC_thres_LSH))
    PQ_total_LSH.append(np.mean(PQ_thres_LSH))
    PC_total_LSH.append(np.mean(PC_thres_LSH))
    F1_total_LSH.append(np.mean(F1_thres_LSH))
    
    FC_total_EC.append(np.mean(FC_thres_EC))
    PQ_total_EC.append(np.mean(PQ_thres_EC))
    PC_total_EC.append(np.mean(PC_thres_EC))
    F1_total_EC.append(np.mean(F1_thres_EC))


#%% make plots


fig1, ax1 = plt.subplots(1)
ax1.plot(FC_total_LSH, PC_total_LSH)
ax1.set_title('LSH')
ax1.set_xlabel('Fraction of comparisons')
ax1.set_ylabel('Pair completeness')
plt.show()

fig2, ax2 = plt.subplots(1)
ax2.plot(FC_total_LSH, PQ_total_LSH)
ax2.set_title('LSH')
ax2.set_xlabel('Fraction of comparisons')
ax2.set_ylabel('Pair quality')
plt.show()

fig3, ax3 = plt.subplots(1)
ax3.plot(FC_total_LSH, F1_total_LSH)
ax3.set_title('LSH')
ax3.set_xlabel('Fraction of comparisons')
ax3.set_ylabel('F1-measure')
plt.show()

fig4, ax4 = plt.subplots(1)
ax4.plot(FC_total_EC, PC_total_EC)
ax4.set_title('EC')
ax4.set_xlabel('Fraction of comparisons')
ax4.set_ylabel('Pair completeness')
plt.show()

fig5, ax5 = plt.subplots(1)
ax5.plot(FC_total_EC, PQ_total_EC)
ax5.set_title('EC')
ax5.set_xlabel('Fraction of comparisons')
ax5.set_ylabel('Pair quality')
plt.show()

fig6, ax6 = plt.subplots(1)
ax6.plot(FC_total_EC, F1_total_EC)
ax6.set_title('EC')
ax6.set_xlabel('Fraction of comparisons')
ax6.set_ylabel('F1-measure')
plt.show()



    
    
    
    
    