import os
import datetime
import numpy as np
import util_ROC, fixed_settings, util_pos
from corpus_representation2 import *
# from options import Defaults
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from gensim.models import KeyedVectors
from optparse import OptionParser


parser = OptionParser()
parser.add_option("--experiment", type="string", help="Experiment name", default="throwaway")

parser.add_option("--ROC_FILENAME", type="string", help="", default="ROCStories_winter2017.csv")

parser.add_option("--sim_metric", type="string", help="", default="sim1")

parser.add_option("--NUM_STORIES", type="int", help="", default=250)

parser.add_option("--NUM_SAMPLES", type="int", help="", default=100)

(opts, args) = parser.parse_args()


ROC_FILEPATH = os.path.join(fixed_settings.DATA_ROOT, opts.ROC_FILENAME)
opts.ROC_FILEPATH = ROC_FILEPATH

opts.preprocess = True
opts.tfidf_norm = None
opts.tfidf_max_df=.1
opts.tfidf_sublinear_tf = False
opts.tfidf_binary_tf = True
opts.SAMPLES = list(range(opts.NUM_SAMPLES))
opts.SIMILARITY_THRESHOLD = None

assert all([s in range(opts.NUM_STORIES) for s in opts.SAMPLES])

now = datetime.datetime.now()
datetime_str = "%i.%i.%i_%i:%i:%i" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
# fixed_settings.OUT_ROOT = os.path.join(fixed_settings.OUT_ROOT, datetime_str)
fixed_settings.OUT_ROOT = os.path.join(fixed_settings.OUT_ROOT, "%s_%s" % (opts.experiment, datetime_str))
os.mkdir(fixed_settings.OUT_ROOT)

print(opts, "\n")


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
        out += 'Rarest in context: {}'.format(contexts_representation.get_tfidf_wts_of_doc(context_id)) + fixed_settings.NEWLINE
        out += ('Nearest to context: {}'.format(contexts_representation.get_nearest_words_to_doc(context_id))) + fixed_settings.NEWLINE
        out += fixed_settings.NEWLINE

        ######### Find neighbors #########

        # context_rep = contexts_representation.representations[context_id]

        # neighbors_info is a sorted list of all (id, similarity) pairs
        neighbors_info = get_nearest_neighbors(contexts_representation, context_id, completions_representation, opts.sim_metric)

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
            closest_words = [word for (word,score) in completions_representation.get_nearest_words_to_doc(neighbor_id, topn=10)]
            out += 'Nearest words to completion: {}'.format(closest_words) + fixed_settings.NEWLINE

            token_wt_1 = contexts_representation.get_filtered_tfidf_wts_of_doc(context_id)
            token_wt_2 = completions_representation.get_filtered_tfidf_wts_of_doc(neighbor_id)
            _, best_indices = similarity2(token_wt_1, token_wt_2, contexts_representation.word_embeddings_model)

            for idx2, (token2, _) in enumerate(token_wt_2):
              best_idx1 = best_indices[idx2]
              best_word1 = token_wt_1[best_idx1][0]
              out += "{0:20}  {1:20}\n".format(token2, best_word1)

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

    ######### Construct TFIDF model #########
    stories_strings = [" ".join(story) for story in stories]

    tfidf_model = TfidfVectorizer(norm=opts.tfidf_norm, sublinear_tf=opts.tfidf_sublinear_tf, binary=opts.tfidf_binary_tf, max_df=opts.tfidf_max_df, tokenizer=word_tokenize)
    tfidf_model.fit(stories_strings)

    ######### Construct context representations #########
    print("Constructing context reps...")
    if opts.preprocess: contexts_preprocessed = [util_ROC.preprocess(element) for element in contexts]
    context_strings = util_ROC.lump_sentences_into_docs(contexts_preprocessed) # list of strings

    contexts_representation = TextCollection(context_strings, tfidf_model=tfidf_model, word_embeddings_model=word_embeddings_model)

    print("contexts_representation len: ", len(contexts_representation.docs))

    ######### Construct completion representations #########
    print("Constructing completion reps...")
    completion_docs = [[completion] for completion in completions]
    print("preprocessing...")
    if opts.preprocess: completions_preprocessed = [util_ROC.preprocess(element) for element in completion_docs]
    print("lumping...")
    completion_strings = util_ROC.lump_sentences_into_docs(completions_preprocessed) # list of strings

    completions_representation = TextCollection(completion_strings, tfidf_model=tfidf_model, word_embeddings_model=word_embeddings_model)

    print("completions_representation", len(completions_representation.docs))

    print("contexts map: ", len(contexts_representation.tfidf_ids_to_tokens))
    print("completions map: ", len(completions_representation.tfidf_ids_to_tokens))

    print("completion 0: ", completions[0])
    print("completion 0 wts: ", completions_representation.get_tfidf_wts_of_doc(0))

    abi(opts, fixed_settings, stories, contexts_preprocessed, completion_strings, contexts_representation, completions_representation)




if __name__=="__main__":
  main()
