import os
import numpy as np
import util_ROC, fixed_settings, util_pos
from corpus_representation import *
from options import Defaults
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from gensim.models import KeyedVectors


# Options
opts = Defaults()
opts.NUM_STORIES = 2500
# opts.NUM_NEIGHBORS = 10
opts.SIMILARITY_THRESHOLD = None
opts.tfidf_max_df=.1
opts.SAMPLES = list(range(100)) # list(range(250))
print(opts, "\n")

assert all([s in range(opts.NUM_STORIES) for s in opts.SAMPLES])


def abi(opts, fixed_settings, stories, contexts, completions, contexts_representation, completions_representation):
    all_percentiles = []

    for context_id in opts.SAMPLES:
        print("context_id: ", context_id)
        out = ''
        out += '==========================================================================' + fixed_settings.NEWLINE
        out += 'Story' + fixed_settings.NEWLINE
        out += '==========================================================================' + fixed_settings.NEWLINE

        out += '\n'.join(contexts[context_id]) + fixed_settings.NEWLINE  # Print context
        out += "> {}".format(completions[context_id]) + fixed_settings.NEWLINE  # Print last sentence
        out += fixed_settings.NEWLINE
        out += '__About the context__' + fixed_settings.NEWLINE
        out += 'Rarest in context: {}'.format(contexts_representation.get_weights_of_doc(context_id)) + fixed_settings.NEWLINE
        out += ('Nearest to context: {}'.format(contexts_representation.get_nearest_words_to_doc(context_id))) + fixed_settings.NEWLINE
        out += fixed_settings.NEWLINE

        ######### Find neighbors #########

        context_rep = contexts_representation.weighted_docs_embeddings[context_id]

        # neighbors_info is a sorted list of all (id, similarity) pairs
        neighbors_info = completions_representation.get_neighbors_by_embedding_similarity(context_rep, num_neighbors=None, weighted=True)

        neighbors_ids = [doc_id for (doc_id, score) in neighbors_info]

        # print("neighbors_info len: ", len(neighbors_info))
        true_rank = neighbors_ids.index(context_id)
        # print("true rank: ", true_rank)

        out += ('==========================================================================') + fixed_settings.NEWLINE
        out += ('Closest Completions') + fixed_settings.NEWLINE
        out += ('==========================================================================') + fixed_settings.NEWLINE

        context_wts = contexts_representation.docs_tfidf_weights[context_id].toarray()[0]
        print("context_wts: ", len(context_wts))

        out += fixed_settings.NEWLINE
        for rank in range(10):
            (neighbor_id, score) = neighbors_info[rank]
            out += " *** " if neighbor_id==context_id else "     "
            out += ("#{}  {}".format(rank, completions[neighbor_id])) + fixed_settings.NEWLINE  # Print last sentence

            completion_wts = completions_representation.docs_tfidf_weights[neighbor_id].toarray()[0]
            shared_words = [word for (word,score) in get_shared_weights(context_wts, completion_wts, contexts_representation.tfidf_ids_to_tokens, completions_representation.tfidf_ids_to_tokens)]


            out += '\nRarest shared words: {}'.format(shared_words) + fixed_settings.NEWLINE
            closest_words = [word for (word,score) in completions_representation.get_nearest_words_to_doc(neighbor_id, weighted=True, topn=10)]
            out += 'Nearest words to completion: {}'.format(closest_words) + fixed_settings.NEWLINE
            out += "\n"

        out += fixed_settings.NEWLINE
        for rank in range(max(0,true_rank-5), min(len(neighbors_info),true_rank+5)):
            (neighbor_id, score) = neighbors_info[rank]
            out += " *** " if neighbor_id==context_id else "     "
            out += ("#{}  {}".format(rank, completions[neighbor_id])) + fixed_settings.NEWLINE  # Print last sentence

        out += fixed_settings.NEWLINE
        for rank in range(len(neighbors_info)-10, len(neighbors_info)):
            (neighbor_id, score) = neighbors_info[rank]
            out += " *** " if neighbor_id==context_id else "     "
            out += ("#{}  {}".format(rank, completions[neighbor_id])) + fixed_settings.NEWLINE  # Print last sentence

        percentile = (opts.NUM_STORIES - 1 - true_rank) * 100.0 / (opts.NUM_STORIES-1)
        all_percentiles.append(percentile)
        out += "\nPercentile rank: %.3f" % percentile

        out_file = os.path.join(fixed_settings.OUT_ROOT, str(context_id))
        print(out, file=open(out_file, 'w'))
        print("Wrote to %s" % out_file)

    out_file = os.path.join(fixed_settings.OUT_ROOT, "results")
    avg_percentile = sum(all_percentiles)/len(all_percentiles)
    out = ""
    for sample,percentile in enumerate(all_percentiles):
      out += "sample %i  percentile %.3f\n" % (sample, percentile)
    result_line = "\nAverage percentile over %i samples and %i candidates: %.3f" % (len(all_percentiles), opts.NUM_STORIES, avg_percentile)
    out += result_line
    print(result_line)
    print(out, file=open(out_file, 'w'))
    # print("Wrote to %s" % out_file)





def main():
    # Setup
    print("Setting up...")
    # stories is list of list of strings
    # contexts is list of list of strings
    # completions is a list of strings
    stories, contexts, completions = util_ROC.get_stories_contexts_and_completions(opts.ROC_FILEPATH, num_stories=opts.NUM_STORIES)
    print("stories len", len(stories), stories[0])
    print("contexts len", len(contexts), contexts[0])
    print("completions len", len(completions), completions[0])
    embedding_filepath = os.path.join(fixed_settings.DATA_ROOT, 'GoogleNews-vectors-negative300.bin')
    word_embeddings_model = KeyedVectors.load_word2vec_format(embedding_filepath, binary=True, limit=20000)

    ######### Construct context representations #########
    print("Constructing context reps...")
    if opts.preprocess: contexts_preprocessed = [util_ROC.preprocess(element) for element in contexts]
    context_strings = util_ROC.lump_sentences_into_docs(contexts_preprocessed) # list of strings
    contexts_tfidf_model = TfidfVectorizer(norm=opts.tfidf_norm,
                                           sublinear_tf=opts.tfidf_sublinear_tf, binary=opts.tfidf_binary_tf, max_df=opts.tfidf_max_df,
                                           tokenizer=word_tokenize)
    contexts_representation = Corpus_representation(context_strings, tfidf_model=contexts_tfidf_model, word_embeddings_model=word_embeddings_model)

    print("contexts_representation len: ", len(contexts_representation.docs))

    ######### Construct completion representations #########
    print("Constructing completion reps...")
    completion_docs = [[completion] for completion in completions]
    print("preprocessing...")
    if opts.preprocess: completions_preprocessed = [util_ROC.preprocess(element) for element in completion_docs]
    print("lumping...")
    completion_strings = util_ROC.lump_sentences_into_docs(completions_preprocessed) # list of strings
    completions_tfidf_model = TfidfVectorizer(norm=opts.tfidf_norm,
                                              sublinear_tf=opts.tfidf_sublinear_tf, binary=opts.tfidf_binary_tf, max_df=opts.tfidf_max_df,
                                              tokenizer=word_tokenize)
    completions_representation = Corpus_representation(completion_strings, tfidf_model=completions_tfidf_model, word_embeddings_model=word_embeddings_model)

    print("completions_representation", len(completions_representation.docs))

    print("contexts map: ", len(contexts_representation.tfidf_ids_to_tokens))
    print("completions map: ", len(completions_representation.tfidf_ids_to_tokens))

    print("completion 0: ", completions[0])
    print("completion 0 wts: ", completions_representation.get_weights_of_doc(0))

    abi(opts, fixed_settings, stories, contexts_preprocessed, completion_strings, contexts_representation, completions_representation)




if __name__=="__main__":
  main()
