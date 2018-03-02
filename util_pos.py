from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def replace_proper_nouns(sentences):
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [ pos_tag(tokenized_sentence) for tokenized_sentence in tokenized_sentences]

    # for tagged_sentence in tagged_sentences:
    #   for (token,tag) in tagged_sentence:
    #     if tag=="NNPS":
    #       print(token)
    #       print(" ".join([token for (token,_) in tagged_sentence]))


    edited_sentences = [
                        [token if (tag!='NNP' and tag!='NNPS') else 'NNP' for (token,tag) in tagged_sentence]
                        for tagged_sentence in tagged_sentences
                        ]

    # for tokenized_sentence, edited_sentence in zip(tokenized_sentences, edited_sentences):
    #   if tokenized_sentence != edited_sentence:
    #     print(" ".join(tokenized_sentence))
    #     print(" ".join(edited_sentence))
    #     print("")

    edited_sentences = [' '.join(edited_sentence) for edited_sentence in edited_sentences] #Untokenize

    return edited_sentences

def get_verbs(doc):
    verbs = [token for (token,tag) in pos_tag(word_tokenize(doc)) if tag[0]=='V'] # All verb tags
    return verbs
