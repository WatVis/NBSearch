import pandas as pd
import os
import gensim
import numpy as np
import math
import smart_open

# Train and save doc2vec model
def train_doc2vec(vector_size=300, epochs=40):
    print("Start training doc2vec")
    predicted = pd.read_csv(os.getcwd() + '/data/train_rows.csv')
    f= open(os.getcwd() + '/data/corpus.txt',"w+")
    for index, row in predicted.iterrows():
        f.write(row['conc_comment'] + '\n')
    f.close()
    def read_corpus(fname, tokens_only=False):
        with smart_open.open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                tokens = gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    train_corpus = list(read_corpus(os.getcwd() + '/data/corpus.txt'))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(os.getcwd()+'/my_model.doc2vec')
    print("Finish training doc2vec")

# Save search index based on generated saved comments
def doc2vec_search_index(filename='/data/final_comments.csv'):
    the_lan_model = torch.load(os.getcwd()+'/my_model.doc2vec')
    df_rows = pd.read_csv(os.getcwd() + filename)
    df_rows = df_rows.drop(['function_token', 'original_cell_no_comments', 'comment_token'], axis=1)
    emb_vecs = []
    for index, row in df_rows.iterrows():
        emb_vecs.append(bert_model.encode(gensim.utils.simple_preprocess(row['conc_comment'])))
    search_index = create_nmslib_search_index(emb_vecs)
    search_index.saveIndex('doc2vec_search_index.nmslib')