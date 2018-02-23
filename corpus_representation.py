from ROC import util_ROC, fixed_settings, util_misc, util_pos
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import numpy as np
import os
from collections import Counter

class Corpus_representation:
    #TODO: speed up: by only going through the (nonzero) indices of a vec (not all indices of the sparse array)

    def __init__(self, docs, tfidf_model=None, word_embeddings_model=None):
        #TODO for now, do not error check args
        self.docs = docs

        ######### Doc representations #########
        self.docs_tfidf_weights = tfidf_model.fit_transform(docs)
        self.tfidf_ids_to_tokens = {id:token for (token, id) in tfidf_model.vocabulary_.items()}

        self.word_embeddings_model = word_embeddings_model
        self.simple_docs_embeddings, self.weighted_docs_embeddings = self._get_doc_embeddings()
        #TODO replace dicts with lists?

    ######### Get neighbors based on similarity #########

    def get_neighbors_by_tfidf_similarity(self, doc_id, num_neighbors=None, threshold=None):
        if num_neighbors is None and threshold is None: raise TypeError #One of these must have a value
        similarity_metric = self.tfidf_similarity
        similarity_scores = [similarity_metric(doc_id, other_id) for other_id in range(len(self.docs))]
        neighbors = sorted(zip(np.arange(len(similarity_scores)), similarity_scores), key=lambda x:x[1], reverse=True)
        if threshold is not None: # Use threshold to choose neighbors
            return self.threshold_neighbors(neighbors, threshold)
        else:
            return neighbors[1:1+num_neighbors] # Top n neighbors

    def get_neighbors_by_embedding_similarity(self, doc_id, num_neighbors=None, threshold=None, weighted=False):
        if num_neighbors is None and threshold is None: raise TypeError #One of these must have a value
        similarity_metric = self.embedding_similarity
        similarity_scores = [similarity_metric(doc_id, other_id, weighted=weighted) for other_id in range(len(self.docs))]
        neighbors = sorted(zip(np.arange(len(similarity_scores)), similarity_scores), key=lambda x:x[1], reverse=True)
        if threshold is not None: # Use threshold to choose neighbors
            return self.threshold_neighbors(neighbors, threshold)
        else:
            return neighbors[1:1+num_neighbors] # Top n neighbors

    def threshold_neighbors(self, neighbors, threshold):
        num_close_neighbors = 0
        for (neighbor_id,score) in neighbors:
            if score>threshold:
                num_close_neighbors += 1
            else:
                break
        return neighbors[:num_close_neighbors]

    def embedding_similarity(self, doc_id1, doc_id2, weighted=False):
        docs_embeddings = self.weighted_docs_embeddings if weighted else self.simple_docs_embeddings
        return util_misc.cosine_similarity(docs_embeddings[doc_id1], docs_embeddings[doc_id2])

    def tfidf_similarity(self, doc_id1, doc_id2):
        return util_misc.cosine_similarity(self.docs_tfidf_weights[doc_id1].toarray()[0], self.docs_tfidf_weights[doc_id2].toarray()[0])

    ######### Get core of neighborhood #########

    def get_nearest_words_to_neighborhood(self, neighbors_info, topn=10, weighted=False):
        docs_embeddings = self.weighted_docs_embeddings if weighted else self.simple_docs_embeddings
        doc_neighborhood_embedding = sum([docs_embeddings[neighbor_id] for (neighbor_id, score) in neighbors_info]) / len(neighbors_info)
        words = self.word_embeddings_model.most_similar(positive=[doc_neighborhood_embedding], topn=topn, restrict_vocab=20000)
        return [(word, self.word_embeddings_model.wv.vocab[word].index, self.word_embeddings_model.wv.vocab[word].count, similarity)
                    for (word, similarity) in words]

    def get_nearest_words_to_neighborhood_keywords(self, neighbors_info, topn=10):
        keywords_embs = [self.get_word_embedding(word) for (neighbor_id, score) in neighbors_info for (word,score) in self.get_nearest_words_to_doc(neighbor_id)]
        neighborhood_emb = sum(keywords_embs)
        words = self.word_embeddings_model.most_similar(positive=[neighborhood_emb], topn=topn, restrict_vocab=20000)
        return [(word, self.word_embeddings_model.wv.vocab[word].index, self.word_embeddings_model.wv.vocab[word].count, similarity)
                    for (word, similarity) in words]

    # def get_most_common_words_in_neighborhood_keywords(self, neighbors_info, topn=10):
    #     tokenized_neighbors = [word_tokenize(self.docs[neighbor_id]) for (neighbor_id, _) in neighbors_info]
    #     tokenized_neighborhood = [token for tokenized_doc in tokenized_neighbors for token in tokenized_doc]
    #     return Counter(tokenized_neighborhood).most_common(topn)

    def get_most_common_words_in_neighborhood(self, neighbors_info, topn=10):
        tokenized_neighbors = [word_tokenize(self.docs[neighbor_id]) for (neighbor_id, _) in neighbors_info]
        tokenized_neighborhood = [token for tokenized_doc in tokenized_neighbors for token in tokenized_doc]
        return Counter(tokenized_neighborhood).most_common(topn)

    def get_nearest_words_to_neighborhood_verbs(self, neighborhood_verbs, topn=10, weighted=False):
        neighborhood_verbs_embedding = sum([self.get_word_embedding(verb) for verb in neighborhood_verbs]) / len(neighborhood_verbs)
        words = self.word_embeddings_model.most_similar(positive=[neighborhood_verbs_embedding], topn=topn, restrict_vocab=20000)
        return [(word, self.word_embeddings_model.wv.vocab[word].index, self.word_embeddings_model.wv.vocab[word].count, similarity)
                    for (word, similarity) in words]
        # return [(word, self.word_embeddings_model.wv.vocab[word].index, self.word_embeddings_model.wv.vocab[word].count,
        #          similarity)
        #         for (word, similarity) in words if word in self.tfidf_ids_to_tokens.values()]

    def get_most_common_verbs_in_neighborhood(self, neighborhood_verbs, topn=10):
        return Counter(neighborhood_verbs).most_common(topn)

    ######### Helpful extra info about a doc #########

    def get_nearest_words_to_doc(self, doc_id, weighted=False, topn=10):
        docs_embeddings = self.weighted_docs_embeddings if weighted else self.simple_docs_embeddings
        doc_embedding = docs_embeddings[doc_id]
        return self.word_embeddings_model.most_similar(positive=[doc_embedding], negative=[], topn=topn, restrict_vocab=20000)

    def get_weights_of_doc(self, doc_id):
        tfidf_weights = self.docs_tfidf_weights[doc_id]
        tokens_to_values = dict(zip([self._tfidf_id_to_token(token_id) for token_id in tfidf_weights.indices], tfidf_weights.data))
        sorted_tokens_to_values = sorted(tokens_to_values.items(), key=lambda pair: pair[1], reverse=True)
        return sorted_tokens_to_values

    def get_weights_shared_between_docs(self, doc_id1, doc_id2):
        tokens_to_values = dict()
        column = 0 #Track column
        for (token_value1, token_value2) in zip(self.docs_tfidf_weights[doc_id1].toarray()[0], self.docs_tfidf_weights[doc_id2].toarray()[0]):
            if token_value1!=0 and token_value2!=0: #Shared word
                token = self._tfidf_id_to_token(column)
                tokens_to_values[token] = (token_value1,token_value2)
            column += 1
        sorted_tokens_to_values = sorted(tokens_to_values.items(), key=lambda pair: pair[1][0], reverse=True)
        return sorted_tokens_to_values

    ######### Helper functions #########

    def _tfidf_id_to_token(self, token_id):
        return self.tfidf_ids_to_tokens[token_id]

    def _get_doc_embedding(self, doc_id):
        simple_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        weighted_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        weights = self.docs_tfidf_weights[doc_id]
        for (tfidf_token_id, weight) in zip(weights.indices, weights.data):  # Go through weights
            token = self._tfidf_id_to_token(tfidf_token_id)  # Get word
            try:
                word_embedding = 1 * self.get_word_embedding(token)
                weighted_word_embedding = weight * word_embedding
                simple_doc_embedding += word_embedding
                weighted_doc_embedding += weighted_word_embedding
            except KeyError:  # No word embedding found
                pass
        return simple_doc_embedding, weighted_doc_embedding

    def _get_doc_embeddings(self):
        all_simple_doc_embeddings = dict()
        all_weighted_doc_embeddings = dict()
        average_simple_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        average_weighted_doc_embedding = np.zeros(self.word_embeddings_model.vector_size)
        for doc_id in range(len(self.docs)):
            curr_simple_doc_embedding, curr_weighted_doc_embedding = self._get_doc_embedding(doc_id)
            all_simple_doc_embeddings[doc_id] = curr_simple_doc_embedding
            all_weighted_doc_embeddings[doc_id] = curr_weighted_doc_embedding
            average_simple_doc_embedding = average_simple_doc_embedding + curr_simple_doc_embedding
            average_weighted_doc_embedding = average_weighted_doc_embedding + curr_weighted_doc_embedding
        average_simple_doc_embedding /= len(self.docs)
        average_weighted_doc_embedding /= len(self.docs)
        all_simple_doc_embeddings = {doc_id:util_misc.drop(vec,average_simple_doc_embedding) for (doc_id,vec) in all_simple_doc_embeddings.items()}
        all_weighted_doc_embeddings = {doc_id:util_misc.drop(vec,average_weighted_doc_embedding) for (doc_id,vec) in all_weighted_doc_embeddings.items()}
        return all_simple_doc_embeddings, all_weighted_doc_embeddings

    def get_word_embedding(self, token):
        try:
            return self.word_embeddings_model.wv[token]
        except KeyError:
            return np.zeros(self.word_embeddings_model.vector_size)