from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def replace_proper_nouns(sentences):
    tagged_sentences = [ pos_tag(word_tokenize(sentence)) for sentence in sentences]
    edited_sentences = [
                        [token if (tag!='NNP' and tag!='NNPS') else 'NNP' for (token,tag) in tagged_sentence]
                        for tagged_sentence in tagged_sentences
                        ]
    edited_sentences = [' '.join(edited_sentence) for edited_sentence in edited_sentences] #Untokenize
    return edited_sentences

def get_verbs(doc):
    verbs = [token for (token,tag) in pos_tag(word_tokenize(doc)) if tag[0]=='V'] # All verb tags
    return verbs