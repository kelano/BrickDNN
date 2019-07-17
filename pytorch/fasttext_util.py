from torchnlp.word_to_vector import FastText

vectors = FastText()
# Load vectors for any word as a `torch.FloatTensor`
vectors['hello']  # RETURNS: [torch.FloatTensor of size 100]