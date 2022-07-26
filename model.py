#!/usr/bin/env python

"""IsoToYaml.py: Inputs a list of texts in YAML format (id, name, text) and uses a constrained version of kmeans to
create clusters. Adapted from : https://colab.research.google.com/github/dipanjanS/nlp_workshop_odsc19/blob/master/
Module05%20-%20NLP%20Applications/Project02%20-%20Text_Clustering.ipynb#"""

__author__ = "Alexandre Giard"


from k_means_constrained import KMeansConstrained
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import requests
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import csv


stopwords_list = requests.get("https://raw.githubusercontent.com/igorbrigadir/stopwords/master/en/alir3z4.txt").content
stop_words = set(stopwords_list.decode().splitlines())
st = LancasterStemmer()


# Get the data from the stardards formatted in YAML files
def get_data(file_name, type):
    with open(file_name) as file:
        liste_normes = yaml.load(file, Loader=yaml.FullLoader)  # Structure : [{id,name,text},{id...}]
        texts = []
        titles = []
        ids = []
        ISO_TITLE_WEIGHTING = 8  # Keywords in the title are X times more important than those in the text
        NIST_TITLE_WEIGHTING = 5
        for norme in liste_normes:
            formatted_title = " ".join([st.stem(w) for w in word_tokenize(norme['name']) if w not in stop_words])
            if type == 'ISO':
                texts.append((formatted_title + ' ') * ISO_TITLE_WEIGHTING
                             + norme['text'][:1000])  # Cap text size at 1000 characters
            else:
                texts.append(((formatted_title + ' ') * NIST_TITLE_WEIGHTING
                             + norme['text'][:1]).replace('cybersec', 'sec'))  # Cap text size at 1000 characters
            ids.append(norme['id'])
            titles.append(norme['name'])
        return texts, ids, titles


# Retreive data
iso_texts, iso_ids, iso_titles = get_data('ISO_standards.yml', 'ISO')
nist_texts, nist_ids, nist_titles = get_data('NIST_standards.yml', 'NIST')
clusters = []  # List of tuples : [C1(standards, features), (standards, features)]

for k in range(len(iso_texts)):
    texts, ids = [iso_texts[k]] + nist_texts, [iso_ids[k]] + nist_ids

    # TF-IDF considering unigrams and bigrams
    cv = CountVectorizer(ngram_range=(1, 2), min_df=0.001, max_df=0.15)  # max_df set to eliminate most common words (like 'sec')
    cv_matrix = cv.fit_transform(texts).todense()

    # Create Kmeans model
    NUM_CLUSTERS, MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE = 28, 2, 4
    model = KMeansConstrained(n_clusters=NUM_CLUSTERS, size_min=MIN_CLUSTER_SIZE, size_max=MAX_CLUSTER_SIZE, max_iter=1000,
                              n_init=50)  # Initializing KMeans model
    km = model.fit(cv_matrix)  # Applying it to our data

    # Get top 4 key features for each cluster
    TOPN_FEATURES = 4
    key_features = []
    feature_names = cv.get_feature_names_out()
    ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for cluster_num in range(NUM_CLUSTERS):
        key_features.append([feature_names[index] for index in ordered_centroids[cluster_num, :TOPN_FEATURES]])

    # Print results
    clustered_texts = [[] for i in range(NUM_CLUSTERS)]
    for i in range(len(ids)):
        clustered_texts[km.labels_[i]].append(ids[i])
    for i in range(len(clustered_texts)):
        for standard_id in clustered_texts[i]:
            if standard_id in iso_ids:
                clusters.append((clustered_texts[i], key_features[i]))
                print((clustered_texts[i], key_features[i]))
                break

header = ['ISO27002 standards', 'NIST Cybersecurity Framework standards', 'Common keywords']
with open('mapping2.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f, delimiter=";", lineterminator='\n')
    writer.writerow(header)  # write the header
    for i in range(len(iso_ids)):  # write the data
        nist_standards = []
        for nist__id in clusters[i][0][1:]:
            nist_standards.append(nist__id + ":" + nist_titles[nist_ids.index(nist__id)])
        data = [clusters[i][0][0] + ":" + iso_titles[i]] + [", ".join(nist_standards)] + [clusters[i][1][1:]]
        writer.writerow(data)
