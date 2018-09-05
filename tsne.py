from dataset.dataset import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from dataset.bow import *


def build_dataset():
    theme_bow_vec = get_bow_vecs('Theme')
    set_name_bow_vec = get_bow_vecs('Name', split=True, stem=True)

    # X_cols = list(range(0, len(set_name_bow_vec)))
    X_cols = list(range(0, 20))
    # X_cols = [0, 1] + range(2, len(theme_bow_vec) + 2)
    # Y_col = list(range(len(set_name_bow_vec), len(set_name_bow_vec) + len(theme_bow_vec)))
    Y_col = list(range(20, 20 + 1))
    # Y_col = list(range(len(set_name_bow_vec), len(set_name_bow_vec) + 2))

    feature_augment_dict = {
    }

    # ['Number', 'Theme', 'Subtheme', 'Year', 'SetName', 'Minifigs', 'Pieces',
    # 'UKPrice', 'USPrice', 'CAPrice', 'EAN', 'UPC', 'Notes', 'QtyOwned', 'NewValue(USD)', 'UsedValue(USD)']
    feats = ["Name", "Theme"]


    import fasttext
    print('loading fasttest')
    word_2_vec, word_to_idx, idx_to_vec, tf_embedding = fasttext.load_vectors()
    print('done loading')

    # set_name_bow_vec_2 = get_bow_vecs('Name', split=True, stem=False)
    # print len(set_name_bow_vec)
    # print len(set_name_bow_vec_2)


    feature_converter_dict = {
        "Theme": lambda str_value: list((int(str_value != 'Star Wars'), int(str_value == 'Star Wars'))),
        # "Theme": lambda str_value: get_bow_encoding(theme_bow_vec, str_value),
        # "Name": lambda str_value: get_bow_encoding(set_name_bow_vec, str_value, split=True, stem=True),
        "Name": lambda str_value: fasttext.get_fasttext_tokenization(word_to_idx, str_value, pad_to=20)
    }

    feats_to_cols = {
        # "Name": list(range(0, len(set_name_bow_vec)))
        "Name": list(range(0, 300))
        # "Theme": range(len(set_name_bow_vec), len(set_name_bow_vec) + len(theme_bow_vec))
    }

    np_dataset = get_datasets(feats, feature_converter_dict, X_cols, Y_col)
    # np_dataset = np_dataset[0:100, :]

    # tot = np.zeros(len(theme_bow_vec))
    # for np_data in np_dataset:
    #     tot += np_data[feats_to_cols['Theme']]

    train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_dataset(np_dataset, X_cols, Y_col)

    save_dataset(train_X, train_Y, dev_X, dev_Y, test_X, test_Y, 'tsnedataset')
    # quit()

    return word_2_vec, word_to_idx, idx_to_vec, tf_embedding


def tsne_plot(X, Y, word_2_vec, word_to_idx, idx_to_vec, tf_embedding):
    idx_to_word = {}
    for word in word_to_idx:
        idx_to_word[word_to_idx[word]] = word

    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    colors = []

    for idx in range(0, len(X)):
        x = X[idx]
        y = Y[idx]

        for word_idx in x:
            tokens.append(idx_to_vec[word_idx])
            labels.append(idx_to_word[word_idx])
            colors.append('r' if y == 0 else 'g')

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=250, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=colors)
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


word_2_vec, word_to_idx, idx_to_vec, tf_embedding = build_dataset()

train_X, train_Y, dev_X, dev_Y, test_X, test_Y = load_dataset(name='tsnedataset')

tsne_plot(train_X, train_Y, word_2_vec, word_to_idx, idx_to_vec, tf_embedding)

print(len(train_Y))
