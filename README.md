# NBSearch

*NBSearch* is a system that supports semantic code search in Jupyter Notebook collections and interactive visual exploration of search results. Details about the system can be found in the following publication: 

>Xingjun Li, Yuanxin Wang, Hong Wang, Yang Wang, and Jian Zhao. [NBSearch: Semantic Search and Visual Exploration ofComputational Notebooks.](https://www.jeffjianzhao.com/papers/nbsearch.pdf) In Proceedings of the ACM SIGCHI Conference on Human Factors in Computing Systems, 2021. 

A video demo of NBSearch can be viewed [here](https://youtu.be/wNSbivrYc0Y).

This repo corresponds to building the backend search engine of NBSearch. `end2end/` includes the Python code for training and testing the models for the search engine. `exploration/` includes the Jupyter Notebooks for some exploratory analysis and usages of the trained models.

## Dataset

A notebook collection can be downloaded from the [UCSD repository](https://library.ucsd.edu/dc/object/bb2733859v). The dataset needs to be placed in `data/`.

## Training Process

The raw Jupyter Notebooks should be in `data/notebooks` after extracting from the UCSD dataset. Go to `end2end/` and follow the below procedure to train and test the translation and language models for the semantic search engine. 

### Create Environment

conda env create -f environment.yml

### Run

python main.py

### Preprocess Text

preprocessing('data/path')

### Train and Test Models

model = Seq2SeqModel(model_option='lstm')

model.create_model()

model.train_model(batch_size=120, epochs=30)

model.evaluate_model(nums=100)

### Predict Comments from Code

model.predict_seq2seq_model(filename='/data/final_comments.csv')

* options of model name are 'gru', 'lstm', 'bilstm', 'lstmattention'

* default model.evaluate_model() will test all data from the testing file generated by preprocess
