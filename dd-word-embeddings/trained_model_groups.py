_Prodv100_loc = 's3://bluetrain-workspaces/kelleng/dd-data/trained-models/Prodv100/%s'
_Prodv104_loc = 's3://bluetrain-workspaces/kelleng/dd-data/trained-models/Prodv104/%s'
_Prodv107_loc = 's3://bluetrain-workspaces/kelleng/dd-data/trained-models/Prodv107/%s'
_Prodv100_embedding = 's3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset.vec'
_Prodv104_embedding = 's3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset-v104.vec'
_Prodv107_embedding = 's3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset-v107.vec'

models = {
    "Prod.v100":
        {
            'BOW-MLP': {
                "loc": _Prodv100_loc % 'BOW-MLP',
                "config": "bow_mlp",
                "embedding": _Prodv100_embedding
            },
            'BOW-MLP-MixedTrain': {
                "loc": _Prodv100_loc % 'BOW-MLP-MixedTrain',
                "config": "bow_mlp",
                "embedding": _Prodv100_embedding
            },
            'SimpleLSTM_150H': {
                "loc": _Prodv100_loc % 'SimpleLSTM_150H',
                "config": "simple_lstm",
                "embedding": _Prodv100_embedding
            },
            'SimpleLSTM_150H_1PAT': {
                "loc": _Prodv100_loc % 'SimpleLSTM_150H_1PAT',
                "config": "simple_lstm",
                "embedding": _Prodv100_embedding
            },
            'SimpleLSTM_150H_2Stack': {
                "loc": _Prodv100_loc % 'SimpleLSTM_150H_2Stack',
                "config": "simple_lstm_2stack",
                "embedding": _Prodv100_embedding
            },
            'SimpleLSTM_150H_05005': {
                "loc": _Prodv100_loc % 'SimpleLSTM_150H_05005',
                "config": "simple_lstm",
                "embedding": _Prodv100_embedding
            },
            'SimpleLSTM_150H_TL': {
                "loc": _Prodv100_loc % 'SimpleLSTM_150H_TL',
                "config": "simple_lstm_TL",
                "embedding": _Prodv100_embedding
            },
            'SimpleLSTM_150H_TL_FT': {
                "loc": _Prodv100_loc % 'SimpleLSTM_150H_TL_FT',
                "config": "simple_lstm_TL_FT",
                "embedding": _Prodv100_embedding
            },
        },
    "Prod.v104":
        {
            'BOW-MLP': {
                "loc": _Prodv104_loc % 'BOWMLPv2_600',
                "config": "bow_mlp",
                "embedding": _Prodv104_embedding
            },
            'BOW-MLP_150H': {
                "loc": _Prodv104_loc % 'BOWMLPv2_150',
                "config": "bow_mlp_150",
                "embedding": _Prodv104_embedding
            },
            'SimpleLSTM_150H_05005': {
                "loc": _Prodv104_loc % 'SimpleLSTM_150H_05005',
                "config": "simple_lstm_05005",
                "embedding": _Prodv104_embedding
            },
            'SimpleLSTM_150H_05005_MixedTrain': {
                "loc": _Prodv104_loc % 'SimpleLSTM_150H_05005_MixedTrain',
                "config": "simple_lstm_05005",
                "embedding": _Prodv104_embedding
            },
            'SimpleLSTM_150H_TL': {
                "loc": _Prodv104_loc % 'SimpleLSTM_150H_TL',
                "config": "simple_lstm_TL",
                "embedding": _Prodv104_embedding
            },
            'SimpleLSTM_150H_TL_FT': {
                "loc": _Prodv104_loc % 'SimpleLSTM_150H_TL_FT',
                "config": "simple_lstm_TL_FT",
                "embedding": _Prodv104_embedding
            },
            'SimpleLSTM_150H_2Stack_TL': {
                "loc": _Prodv104_loc % 'SimpleLSTM_150H_2Stack_TL',
                "config": "simple_lstm_2stack_TL",
                "embedding": _Prodv104_embedding
            },
            'SimpleLSTM_150H_2Stack_TL_FT': {
                "loc": _Prodv104_loc % 'SimpleLSTM_150H_2Stack_TL_FT',
                "config": "simple_lstm_2stack_TL_FT",
                "embedding": _Prodv104_embedding
            },
            'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw': {
                "loc": _Prodv104_loc % 'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw',
                "config": "simple_lstm_2stack_TL_FT_HalfThaw",
                "embedding": _Prodv104_embedding
            },
        },
    "Prod.v107":
        {
            'SimpleLSTM_150H_2Stack_TL': {
                "loc": _Prodv107_loc % 'SimpleLSTM_150H_2Stack_TL',
                "config": "simple_lstm_2stack_TL",
                "embedding": _Prodv107_embedding
            },
            'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw': {
                "loc": _Prodv107_loc % 'SimpleLSTM_150H_2Stack_TL_FT_HalfThaw',
                "config": "simple_lstm_2stack_TL_FT_HalfThaw",
                "embedding": _Prodv107_embedding
            },
        },
}
