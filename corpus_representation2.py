import util_ROC, fixed_settings, util_misc, util_pos
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import numpy as np
import os
from collections import Counter

class TextCollection:
    """Holds a set of texts, and representations for each member in the set."""

    def __init__(self, docs, tfidf_model=None, word_embeddings_model=None):
        self.docs = docs # is list of list of strings

        ######### Doc representations #########
        self.docs_tfidf_weights = tfidf_model.fit_transform(docs) # this is a matrix shape (vocab size, num docs) or similar

        self.tfidf_ids_to_tokens = {id:token for (token, id) in tfidf_model.vocabulary_.items()} # a vocabulary map

        self.word_embeddings_model = word_embeddings_model # a gensim word2vec model

        self.simple_docs_embeddings, self.weighted_docs_embeddings = self.init_doc_embeddings()


    def get_nearest_words_to_doc(self, doc_id, topn=10):
        doc_embedding = self.weighted_docs_embeddings[doc_id]
        return self.word_embeddings_model.most_similar(positive=[doc_embedding], negative=[], topn=topn, restrict_vocab=20000)

    def get_tfidf_wts_of_doc(self, doc_id):
        """Returns list of (token, tfidf weight) i.e. (string, float) pairs, sorted by tfidf weight"""
        tfidf_weights = self.docs_tfidf_weights[doc_id] # length vocab size

        tokens_to_values = zip([self.tfidf_ids_to_tokens[token_id] for token_id in tfidf_weights.indices], tfidf_weights.data)

        tokens_wts = sorted(tokens_to_values, key=lambda pair: pair[1], reverse=True)

        if len(tokens_wts)==0:
          print("Zero length tfidf weights warning: doc_id=%i, doc=%s" % (doc_id, self.docs[doc_id]))

        return tokens_wts

    def get_filtered_tfidf_wts_of_doc(self, doc_id):
      """Same as get_tfidf_wts_of_doc but removes entries that have no word embedding"""
      tokens_wts = self.get_tfidf_wts_of_doc(doc_id)
      tokens_wts_filtered = [(token, wt) for (token, wt) in tokens_wts if token in self.word_embeddings_model]
      if len(tokens_wts_filtered)==0:
        print("Zero length filtered tfidf weights warning: doc_id=%i, doc=%s" % (doc_id, self.docs[doc_id]))
      return tokens_wts_filtered


    # def get_tfidf_and_wordvecs_of_doc(self, doc_id):
    #     """Returns list of (token, tfidf weight, embedding) triples, sorted by tfidf weight"""
    #     tfidf_wts = self.get_tfidf_wts_of_doc(doc_id) # sorted list of (token, tfidf wt) pairs
    #     token_wt_emb = []
    #
    #     for token, wt in tfidf_wts:
    #         try:
    #             wordemb = self.word_embeddings_model.word_vec(token)
    #         except KeyError:
    #             continue
    #         token_wt_emb.append((token, wt, wordemb))
    #
    #     return token_wt_emb

    def get_weights_shared_between_docs(self, doc_id1, doc_id2):
        tokens_to_values = dict()
        column = 0 #Track column
        for (token_value1, token_value2) in zip(self.docs_tfidf_weights[doc_id1].toarray()[0], self.docs_tfidf_weights[doc_id2].toarray()[0]):
            if token_value1!=0 and token_value2!=0: #Shared word
                token = self.tfidf_ids_to_tokens[column]
                tokens_to_values[token] = (token_value1,token_value2)
            column += 1
        sorted_tokens_to_values = sorted(tokens_to_values.items(), key=lambda pair: pair[1][0], reverse=True)
        return sorted_tokens_to_values

    ######### Helper functions #########

    def init_doc_embedding(self, doc_id):
        simple_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        weighted_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        weights = self.docs_tfidf_weights[doc_id]
        for (tfidf_token_id, weight) in zip(weights.indices, weights.data):  # Go through weights
            token = self.tfidf_ids_to_tokens[tfidf_token_id]  # Get word
            try:
                word_embedding = 1 * self.word_embeddings_model.word_vec(token)
                weighted_word_embedding = weight * word_embedding
                simple_doc_embedding += word_embedding
                weighted_doc_embedding += weighted_word_embedding
            except KeyError:  # No word embedding found
                pass
        return simple_doc_embedding, weighted_doc_embedding

    def init_doc_embeddings(self):
        """Calculate embedding for each doc.
        Project out average embedding from each embedding."""

        all_simple_doc_embeddings = dict()
        all_weighted_doc_embeddings = dict()

        average_simple_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        average_weighted_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        for doc_id in range(len(self.docs)):
            curr_simple_doc_embedding, curr_weighted_doc_embedding = self.init_doc_embedding(doc_id)
            all_simple_doc_embeddings[doc_id] = curr_simple_doc_embedding
            all_weighted_doc_embeddings[doc_id] = curr_weighted_doc_embedding
            average_simple_doc_embedding = average_simple_doc_embedding + curr_simple_doc_embedding
            average_weighted_doc_embedding = average_weighted_doc_embedding + curr_weighted_doc_embedding

        average_simple_doc_embedding /= len(self.docs)
        average_weighted_doc_embedding /= len(self.docs)

        # project out the average embedding
        all_simple_doc_embeddings = {doc_id:util_misc.drop(vec,average_simple_doc_embedding) for (doc_id,vec) in all_simple_doc_embeddings.items()}
        all_weighted_doc_embeddings = {doc_id:util_misc.drop(vec,average_weighted_doc_embedding) for (doc_id,vec) in all_weighted_doc_embeddings.items()}
        return all_simple_doc_embeddings, all_weighted_doc_embeddings


def get_shared_weights(wts1, wts2, mapping1, mapping2):
    wts_tokens1 = {mapping1[token_id]:wt for token_id,wt in enumerate(wts1) if wt!=0}
    wts_tokens2 = {mapping2[token_id]:wt for token_id,wt in enumerate(wts2) if wt!=0}

    # print("wts_tokens1", wts_tokens1)
    # print("wts_tokens2", wts_tokens2)

    tokens_to_values = dict()

    for token, wt1 in wts_tokens1.items():
      if token in wts_tokens2:
        tokens_to_values[token] = (wt1, wts_tokens2[token])

    sorted_tokens_to_values = sorted(tokens_to_values.items(), key=lambda pair: pair[1][0], reverse=True)
    # print("sorted_tokens_to_values: ", sorted_tokens_to_values)
    return sorted_tokens_to_values


def get_nearest_neighbors(tc1, doc_id1, tc2, sim_metric):
  """Get nearest neighbors in tc2 to doc_id1 (in tc1)"""
  token_wt_1 = tc1.get_filtered_tfidf_wts_of_doc(doc_id1)
  tc2_len = len(tc2.docs)

  if sim_metric=="sim1":
    ######## SIMILARITY 1 #########

    # list of scores, in same order as self.docs
    sim_scores = []
    for doc_id2 in range(tc2_len):
      sim = similarity1(tc1, tc2, doc_id1, doc_id2)
      sim_scores.append(sim)

    # neighbors is list of (neighbor_id, score) pairs, sorted w.r.t. score, and the neighbor_id is the place it appears in tc2.docs
    neighbors = sorted(zip(np.arange(len(sim_scores)), sim_scores), key=lambda x:x[1], reverse=True)

  elif sim_metric=="sim1_prime":
    ######## SIMILARITY 1 PRIME #########

    # list of scores, in same order as self.docs
    sim_scores = []
    for doc_id2 in range(tc2_len):
      token_wt_2 = tc2.get_filtered_tfidf_wts_of_doc(doc_id2)
      sim = similarity1_prime(token_wt_1, token_wt_2, tc1.word_embeddings_model)
      sim_scores.append(sim)

    # neighbors is list of (neighbor_id, score) pairs, sorted w.r.t. score, and the neighbor_id is the place it appears in tc2.docs
    neighbors = sorted(zip(np.arange(len(sim_scores)), sim_scores), key=lambda x:x[1], reverse=True)

  elif sim_metric=="sim2":
    ######### SIMILARITY 2 #########

    # list of scores, in same order as self.docs
    sim_scores = []
    for doc_id2 in range(tc2_len):
      token_wt_2 = tc2.get_filtered_tfidf_wts_of_doc(doc_id2)
      sim, _ = similarity2(token_wt_1, token_wt_2, tc1.word_embeddings_model)
      sim_scores.append(sim)

    # neighbors is list of (neighbor_id, score) pairs, sorted w.r.t. score, and the neighbor_id is the place it appears in tc2.docs
    neighbors = sorted(zip(np.arange(len(sim_scores)), sim_scores), key=lambda x:x[1], reverse=True)

  elif sim_metric=="sim3":
    emb1 = tc1.weighted_docs_embeddings[doc_id1]

    # list of scores, in same order as self.docs
    sim_scores = []
    for doc_id2 in range(tc2_len):
      token_wt_2 = tc2.get_filtered_tfidf_wts_of_doc(doc_id2)
      sim = similarity3(token_wt_2, emb1, tc1.word_embeddings_model)
      sim_scores.append(sim)

    # neighbors is list of (neighbor_id, score) pairs, sorted w.r.t. score, and the neighbor_id is the place it appears in tc2.docs
    neighbors = sorted(zip(np.arange(len(sim_scores)), sim_scores), key=lambda x:x[1], reverse=True)

  elif sim_metric=="sim4":
    ######### SIMILARITY 2 #########

    # list of scores, in same order as self.docs
    sim_scores = []
    for doc_id2 in range(tc2_len):
      token_wt_2 = tc2.get_filtered_tfidf_wts_of_doc(doc_id2)
      sim, _ = similarity4(token_wt_1, token_wt_2, tc1.word_embeddings_model)
      sim_scores.append(sim)

    # neighbors is list of (neighbor_id, score) pairs, sorted w.r.t. score, and the neighbor_id is the place it appears in tc2.docs
    neighbors = sorted(zip(np.arange(len(sim_scores)), sim_scores), key=lambda x:x[1], reverse=True)


  else:
    raise ValueError("Unexpected values of sim_metric: ", sim_metric)

  return neighbors



def similarity1(tc1, tc2, doc_id1, doc_id2):
    """Returns similarity 1 which is just cosine dist of weighted embeddings"""
    emb1 = tc1.weighted_docs_embeddings[doc_id1]
    emb2 = tc2.weighted_docs_embeddings[doc_id2]
    return util_misc.cosine_similarity(emb1, emb2)


def similarity1_prime(token_wt_1, token_wt_2, word_embeddings_model):
    """Returns similarity 1, which is like similarity1 but without projecting out the average"""

    len1 = len(token_wt_1)
    len2 = len(token_wt_2)

    if len1==0 or len2==0:
      print("Warning: assigning similarity 0 because len=0")
      return 0

    # calc weighted embeddings
    emb1 = sum([wt*word_embeddings_model.word_vec(token) for (token,wt) in token_wt_1])
    emb2 = sum([wt*word_embeddings_model.word_vec(token) for (token,wt) in token_wt_2])

    sim = util_misc.cosine_similarity(emb1, emb2)

    return sim

def similarity2(token_wt_1, token_wt_2, word_embeddings_model):
    """Returns similarity 2, which is like distance 2 but similarity instead of distance"""

    len1 = len(token_wt_1)
    len2 = len(token_wt_2)

    # print("len1, len2: ", len1, len2)
    # print("tokens1: ", " ".join([token1 for (token1, _) in token_wt_1]))
    # print("tokens2: ", " ".join([token2 for (token2, _) in token_wt_2]))
    # print("token_wt_1: ", token_wt_1)
    # print("token_wt_2: ", token_wt_2)

    # print("calculating alignments...")
    alignments = np.zeros((len1, len2)) # contains DISTANCES
    for idx1, (token1, wt1) in enumerate(token_wt_1):
      for idx2, (token2, wt2) in enumerate(token_wt_2):
        # alignments[idx1, idx2] = util_misc.cosine_similarity(word_embeddings_model.word_vec(token1), word_embeddings_model.word_vec(token2))
        alignments[idx1, idx2] = word_embeddings_model.similarity(token1, token2)

    # print("alignments: ")
    # for row in alignments:
    #   print(row)

    # print("taking max...")
    best_sims = np.amax(alignments, axis=0) # array length len2
    best_indices = np.argmax(alignments, axis=0) # array length len2

    # print("best_indices: ", best_indices)
    for idx2, (token2, _) in enumerate(token_wt_2):
      best_idx1 = best_indices[idx2]
      best_word1 = token_wt_1[best_idx1][0]
      # print("{0:20}  {1:20}".format(token2, best_word1))

    # print("best_dists: ", best_dists)

    # print("computing distance...")
    sim = 0
    for idx2, (token2, wt2) in enumerate(token_wt_2):
      sim += wt2 * best_sims[idx2]
    sim /= len2

    # print("dist2: ", dist)
    # print("")

    return sim, best_indices


def similarity3(token_wt_2, emb1, word_embeddings_model):

    len2 = len(token_wt_2)

    sim = 0
    for (token2, wt2) in token_wt_2:
      emb2 = word_embeddings_model.word_vec(token2)
      sim += wt2 * util_misc.cosine_similarity(emb2, emb1)
    sim /= len2

    return sim

def similarity4(token_wt_1, token_wt_2, word_embeddings_model):
    """Returns similarity 2, which is like distance 2 but similarity instead of distance"""

    len1 = len(token_wt_1)
    len2 = len(token_wt_2)

    # print("len1, len2: ", len1, len2)
    # print("tokens1: ", " ".join([token1 for (token1, _) in token_wt_1]))
    # print("tokens2: ", " ".join([token2 for (token2, _) in token_wt_2]))
    # print("token_wt_1: ", token_wt_1)
    # print("token_wt_2: ", token_wt_2)

    # print("calculating alignments...")
    alignments = np.zeros((len1, len2)) # contains DISTANCES
    for idx1, (token1, wt1) in enumerate(token_wt_1):
      for idx2, (token2, wt2) in enumerate(token_wt_2):
        # alignments[idx1, idx2] = util_misc.cosine_similarity(word_embeddings_model.word_vec(token1), word_embeddings_model.word_vec(token2))
        alignments[idx1, idx2] = wt1 * word_embeddings_model.similarity(token1, token2)

    # print("alignments: ")
    # for row in alignments:
    #   print(row)

    # print("taking max...")
    best_sims = np.amax(alignments, axis=0) # array length len2
    best_indices = np.argmax(alignments, axis=0) # array length len2

    # print("best_indices: ", best_indices)
    for idx2, (token2, _) in enumerate(token_wt_2):
      best_idx1 = best_indices[idx2]
      best_word1 = token_wt_1[best_idx1][0]
      # print("{0:20}  {1:20}".format(token2, best_word1))

    # print("best_dists: ", best_dists)

    # print("computing distance...")
    sim = 0
    for idx2, (token2, wt2) in enumerate(token_wt_2):
      sim += wt2 * best_sims[idx2]
    sim /= len2

    # print("dist2: ", dist)
    # print("")

    return sim, best_indices
