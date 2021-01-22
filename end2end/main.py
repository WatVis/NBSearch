from doc2vec import train_doc2vec
from seq2seq import Seq2SeqModel
from preprocess import preprocessing, preprocess_language_model_data

# The path to store notebooks data
path = 'data/notebooks/'

preprocessing(path)

options = ['bilstm'] #, 'gru', 'lstm', 'lstmattention']
for option in options:
    model = Seq2SeqModel(model_option=option)
    model.create_model()
    model.train_model(batch_size=120, epochs=30)
    model.evaluate_model(nums=0,model_option=option)
    
model = Seq2SeqModel(model_option='bilstm')
model.predict_seq2seq_model(filename='/data/final_comments.csv')

preprocess_language_model_data()
train_doc2vec(vector_size=300, epochs=40)
