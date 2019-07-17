from allennlp.commands.elmo import ElmoEmbedder

elmo_embedder = None


def embed(sentence):
    global elmo_embedder
    if elmo_embedder is None:
        elmo_embedder = ElmoEmbedder()
    tokens = sentence.split()
    vectors = elmo_embedder.embed_sentence(tokens)  # (3, sent_len, 1024)
    last_layer = vectors[2]  # (sent_len, 1024)
    return last_layer




# assert(len(vectors) == 3) # one for each layer in the ELMo output
# assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens
#
# import scipy
# vectors2 = elmo.embed_sentence(["I", "ate", "a", "carrot", "for", "breakfast"])
# print(scipy.spatial.distance.cosine(vectors[2][3], vectors2[2][3])) # cosine distance between "apple" and "carrot" in the last layer
#

