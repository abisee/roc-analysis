import os
import numpy as np
from ROC import util_ROC, fixed_settings, util_pos
from ROC.corpus_representation import Corpus_representation
from ROC.options import Defaults
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from gensim.models import KeyedVectors

# Options
opts = Defaults()
opts.NUM_STORIES = 25000
opts.NUM_NEIGHBORS = 10
opts.SIMILARITY_THRESHOLD = None
opts.tfidf_max_df=.1
opts.SAMPLES = list(range(250))
print(opts, "\n")

# Setup
stories, contexts, completions = util_ROC.get_stories_contexts_and_completions(opts.ROC_FILEPATH, num_stories=opts.NUM_STORIES)
embedding_filepath = os.path.join(fixed_settings.DATA_ROOT, 'GoogleNews-vectors-negative300.bin')
word_embeddings_model = KeyedVectors.load_word2vec_format(embedding_filepath, binary=True, limit=20000)

######### Construct context representations #########
docs_sentences = contexts
if opts.preprocess: docs_sentences = [util_ROC.preprocess(element) for element in docs_sentences]
docs = util_ROC.lump_sentences_into_docs(docs_sentences)
contexts_tfidf_model = TfidfVectorizer(norm=opts.tfidf_norm,
                                       sublinear_tf=opts.tfidf_sublinear_tf, binary=opts.tfidf_binary_tf, max_df=opts.tfidf_max_df,
                                       tokenizer=word_tokenize)
contexts_representation = Corpus_representation(docs, tfidf_model=contexts_tfidf_model, word_embeddings_model=word_embeddings_model)

######### Construct completion representations #########
docs_sentences = [[completion] for completion in completions] + contexts
if opts.preprocess: docs_sentences = [util_ROC.preprocess(element) for element in docs_sentences]
docs = util_ROC.lump_sentences_into_docs(docs_sentences)
completions_tfidf_model = TfidfVectorizer(norm=opts.tfidf_norm,
                                          sublinear_tf=opts.tfidf_sublinear_tf, binary=opts.tfidf_binary_tf, max_df=opts.tfidf_max_df,
                                          tokenizer=word_tokenize)
completions_representation = Corpus_representation(docs, tfidf_model=completions_tfidf_model, word_embeddings_model=word_embeddings_model)

for sample in opts.SAMPLES:
    out = ''
    out += '==========================================================================' + fixed_settings.NEWLINE
    out += 'Story' + fixed_settings.NEWLINE
    out += '==========================================================================' + fixed_settings.NEWLINE
    # print("Story #{}".format(sample + 1))
    out += '\n'.join(contexts[sample]) + fixed_settings.NEWLINE  # Print context
    out += "> {}".format(completions[sample]) + fixed_settings.NEWLINE  # Print last sentence
    out += fixed_settings.NEWLINE
    out += '__About the context__' + fixed_settings.NEWLINE
    out += 'Rarest in context: {}'.format(contexts_representation.get_weights_of_doc(sample)) + fixed_settings.NEWLINE
    out += ('Nearest to context: {}'.format(contexts_representation.get_nearest_words_to_doc(sample))) + fixed_settings.NEWLINE
    out += fixed_settings.NEWLINE

    ######### Find neighbors #########
    neighbors_info = contexts_representation.get_neighbors_by_embedding_similarity(sample,
                                                                                       num_neighbors=opts.NUM_NEIGHBORS,
                                                                                       threshold=opts.SIMILARITY_THRESHOLD,
                                                                                        weighted=True)
    out += ('==========================================================================') + fixed_settings.NEWLINE
    out += ('Extracted from Neighbors\' Completions') + fixed_settings.NEWLINE
    out += ('==========================================================================') + fixed_settings.NEWLINE

    ######### Get key of neighborhood completions #########
    #TODO Goal: get (prototype?), seed word, MV, MVP, sentiment, external K
    # print('Most common words in completions: {}'.format(completions_representation.get_most_common_words_in_neighborhood(neighbors_info)))
    out += ('Nearest words to completions: {}'.format(completions_representation.get_nearest_words_to_neighborhood(neighbors_info, weighted=True))) + fixed_settings.NEWLINE
    out += ('Nearest words to completions\' keywords: {}'.format(completions_representation.get_nearest_words_to_neighborhood_keywords(neighbors_info))) + fixed_settings.NEWLINE
    verbs_in_completions = [verb for (neighbor_id, _) in neighbors_info for verb in util_pos.get_verbs(completions[neighbor_id])]
    # print(verbs_in_completions)
    # print('Most common completions\' verbs: {}'.format(completions_representation.get_most_common_verbs_in_neighborhood(verbs_in_completions)))
    out += ('Nearest words to completions\' verbs: {}'.format(completions_representation.get_nearest_words_to_neighborhood_verbs(verbs_in_completions, weighted=True))) + fixed_settings.NEWLINE
    out += fixed_settings.NEWLINE

    out += ('==========================================================================') + fixed_settings.NEWLINE
    out += ('Neighbors') + fixed_settings.NEWLINE
    out += ('==========================================================================') + fixed_settings.NEWLINE
    for (neighbor_id, score) in neighbors_info:
        # neighbor_story = stories[neighbor_id]
        # print("Story #{}".format(neighbor_id + 1))
        out += ('\n'.join(contexts[neighbor_id])) + fixed_settings.NEWLINE  # Print context
        out += ("> {}".format(completions[neighbor_id])) + fixed_settings.NEWLINE  # Print last sentence
        out += fixed_settings.NEWLINE
        out += ('__About the match__') + fixed_settings.NEWLINE
        out += ('Rarest shared words: {}'.format(
            contexts_representation.get_weights_shared_between_docs(sample, neighbor_id))) + fixed_settings.NEWLINE
        out += ("Score: {}".format(score)) + fixed_settings.NEWLINE
        out += fixed_settings.NEWLINE
        out += ('__About the completion__') + fixed_settings.NEWLINE
        out += ('Rarest in completion: {}'.format(completions_representation.get_weights_of_doc(neighbor_id))) + fixed_settings.NEWLINE
        out += ('Nearest to completion: {}'.format(
            completions_representation.get_nearest_words_to_doc(neighbor_id))) + fixed_settings.NEWLINE
        out += fixed_settings.NEWLINE + '--------------------------------------------------------------------------' + fixed_settings.NEWLINE + fixed_settings.NEWLINE

    print(out, file=open(os.path.join(fixed_settings.OUT_ROOT,str(sample+1)), 'a'))
