import dill as dpickle
import logging
import numpy as np
import os
import pandas as pd
import re
import statistics
import tensorflow as tf
import time

from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, TimeDistributed, Attention, RNN, Input, LSTMCell, LSTM, GRU, Dense, \
    Embedding, Bidirectional, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

def load_text_processor(fname='comments_pp.dpkl'):
    """
    Load preprocessors from disk.
    Parameters
    ----------
    fname: str
        file name of ktext.proccessor object
    Returns
    -------
    num_tokens : int
        size of vocabulary loaded into ktext.processor
    pp : ktext.processor
        the processor you are trying to load
    Typical Usage:
    -------------
    num_decoder_tokens, comments_pp = load_text_processor(fname='comments_pp.dpkl')
    num_encoder_tokens, cell_pp = load_text_processor(fname='cell_pp.dpkl')
    """
    # Load files from disk
    with open(fname, 'rb') as f:
        pp = dpickle.load(f)

    num_tokens = max(pp.id2token.keys()) + 1
    print(f'Size of vocabulary for {fname}: {num_tokens:,}')
    return num_tokens, pp


def load_decoder_inputs(decoder_np_vecs='train_comments_vecs.npy'):
    """
    Load decoder inputs.
    Parameters
    ----------
    decoder_np_vecs : str
        filename of serialized numpy.array of decoder input ( comments)
    Returns
    -------
    decoder_input_data : numpy.array
        The data fed to the decoder as input during training for teacher forcing.
        This is the same as `decoder_np_vecs` except the last position.
    decoder_target_data : numpy.array
        The data that the decoder data is trained to generate ( comments).
        Calculated by sliding `decoder_np_vecs` one position forward.
    """
    vectorized_comments = np.load(decoder_np_vecs)
    # For Decoder Input, you don't need the last word as that is only for prediction
    # when we are training using Teacher Forcing.
    decoder_input_data = vectorized_comments[:, :-1]

    # Decoder Target Data Is Ahead By 1 Time Step From Decoder Input Data (Teacher Forcing)
    decoder_target_data = vectorized_comments[:, 1:]

    print(f'Shape of decoder input: {decoder_input_data.shape}')
    print(f'Shape of decoder target: {decoder_target_data.shape}')
    return decoder_input_data, decoder_target_data


def load_encoder_inputs(encoder_np_vecs='train_cell_vecs.npy'):
    """
    Load variables & data that are inputs to encoder.
    Parameters
    ----------
    encoder_np_vecs : str
        filename of serialized numpy.array of encoder input ( comments)
    Returns
    -------
    encoder_input_data : numpy.array
        The  cell
    doc_length : int
        The standard document length of the input for the encoder after padding
        the shape of this array will be (num_examples, doc_length)
    """
    vectorized_cell = np.load(encoder_np_vecs)
    # Encoder input is simply the cell of the  text
    encoder_input_data = vectorized_cell
    doc_length = encoder_input_data.shape[1]
    print(f'Shape of encoder input: {encoder_input_data.shape}')
    return encoder_input_data, doc_length


def extract_encoder_model(model):
    """
    Extract the encoder from the original Sequence to Sequence Model.
    Returns a keras model object that has one input (cell of ) and one
    output (encoding of , which is the last hidden state).
    Input:
    -----
    model: keras model object
    Returns:
    -----
    keras model object
    """
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model


def extract_decoder_model(model, model_option):
    """
    Extract the decoder from the original model.
    Inputs:
    ------
    model: keras model object
    Returns:
    -------
    A Keras model object with the following inputs and outputs:
    Inputs of Keras Model That Is Returned:
    1: the embedding index for the last predicted word or the <Start> indicator
    2: the last hidden state, or in the case of the first word the hidden state from the encoder
    Outputs of Keras Model That Is Returned:
    1.  Prediction (class probabilities) for the next word
    2.  The hidden state of the decoder, to be fed back into the decoder at the next time step
    Implementation Notes:
    ----------------------
    Must extract relevant layers and reconstruct part of the computation graph
    to allow for different inputs as we are not going to use teacher forcing at
    inference time.
    """
    # the latent dimension is the same throughout the architecture so we are going to
    # cheat and grab the latent dimension of the embedding because that is the same as what is
    # output from the decoder
    latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]

    # Reconstruct the input into the decoder
    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)

    # Instead of setting the intial state from the encoder and forgetting about it, during inference
    # we are not doing teacher forcing, so we will have to have a feedback loop from predictions back into
    # the GRU, thus we define this input layer for the state so we can add this capability
    # gru_inference_state_input = Input(shape=(300,), name='hidden_state_input')
    state_h = Input(shape=(300,))
    state_c = Input(shape=(300,))
    encoder_output = Input(shape=(300,))
    # we need to reuse the weights that is why we are getting this
    # If you inspect the decoder GRU that we created for training, it will take as input
    # 2 tensors -> (1) is the embedding layer output for the teacher forcing
    #                  (which will now be the last step's prediction, and will be _start_ on the first time step)
    #              (2) is the state, which we will initialize with the encoder on the first time step, but then
    #                   grab the state after the first prediction and feed that back in again.
    if model_option == 'lstm':
        state_input = [state_h, state_c]
        out, state_h, state_c = model.get_layer('Decoder-LSTM')([dec_bn, state_h, state_c])
        state = [state_h, state_c]
    elif model_option == 'lstmattention':
        state_input = [state_h, state_c, encoder_output]
        o = model.get_layer('Attention')(state_input)
        out, state_h, state_c = model.get_layer('Decoder-LSTM')(
            tf.concat([dec_bn, dec_bn * 0 + tf.expand_dims(o, axis=1)], axis=-1), initial_state=[state_h, state_c])
        state = [state_h, state_c, out]
    elif model_option == 'bilstm':
        state_h = Input(shape=(600,))
        state_c = Input(shape=(600,))
        state_input = [state_h, state_c]
        out, state_h, state_c = model.get_layer('Decoder-LSTM')([dec_bn, state_h, state_c])
        state = [state_h, state_c]
    else:  # GRU
        state_input = [state_h]
        out, state_h = model.get_layer('Decoder-GRU')([dec_bn, state_h])
        state = [state_h]

    # Reconstruct dense layers
    dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(out)
    dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)
    decoder_model = Model([decoder_inputs, state_input],
                          [dense_out] + state)
    # decoder_model = Model([decoder_inputs, gru_inference_state_input],
    #                 [dense_out] + [gru_state_out])
    return decoder_model


class Seq2Seq_Inference(object):
    def __init__(self,
                 encoder_preprocessor,
                 decoder_preprocessor,
                 seq2seq_model, model_option):

        self.pp_cell = encoder_preprocessor
        self.pp_comments = decoder_preprocessor
        self.seq2seq_model = seq2seq_model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.model_option = model_option.lower()
        self.decoder_model = extract_decoder_model(seq2seq_model, model_option)
        self.default_max_len_comments = self.pp_comments.padding_maxlen
        self.nn = None
        self.rec_df = None

    def generate_comments(self,
                          raw_input_text,
                          max_len_comments=None):
        """
        Use the seq2seq model to generate a comments given the cell of an .
        Inputs
        ------
        raw_input: str
            The cell of the  text as an input string
        max_len_comments: int (optional)
            The maximum length of the comments the model will generate
        """
        if max_len_comments is None:
            max_len_comments = self.default_max_len_comments
        # get the encoder's features for the decoder
        raw_tokenized = self.pp_cell.transform([raw_input_text])
        cell_encoding = self.encoder_model.predict(raw_tokenized)
        # we want to save the encoder's embedding before its updated by decoder
        #   because we can use that as an embedding for other tasks.
        original_cell_encoding = cell_encoding
        state_value = np.array(self.pp_comments.token2id['_start_']).reshape(1, 1)
        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            outputs = self.decoder_model.predict([state_value, cell_encoding])
            preds = outputs[0]
            # We are going to ignore indices 0 (padding) and indices 1 (unknown)
            # Argmax will return the integer index corresponding to the
            #  prediction + 2 b/c we chopped off first two
            pred_idx = np.argmax(preds[:, :, 2:]) + 2

            # retrieve word from index prediction
            pred_word_str = self.pp_comments.id2token[pred_idx]

            if pred_word_str == '_end_' or len(decoded_sentence) >= max_len_comments:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)
            # update the decoder for the next word
            if self.model_option == 'lstmattention':
                cell_encoding = [outputs[1], outputs[2], np.squeeze(outputs[3], axis=(1,))]
            elif self.model_option == 'lstm':
                cell_encoding = [outputs[1], outputs[2]]
            elif self.model_option == 'bilstm':
                cell_encoding = [outputs[1], outputs[2]]
            else:
                cell_encoding = [outputs[1]]
            state_value = np.array(pred_idx).reshape(1, 1)

        return original_cell_encoding, ' '.join(decoded_sentence)


class Seq2SeqModel(object):
    def __init__(self, model_option='gru'):
        # from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor
        self.encoder_input_data, self.doc_length = load_encoder_inputs(os.getcwd() + '/data/train_cell_vecs.npy')
        self.decoder_input_data, self.decoder_target_data = load_decoder_inputs(
            os.getcwd() + '/data/train_comments_vecs.npy')
        self.num_encoder_tokens, self.cell_pp = load_text_processor(os.getcwd() + '/data/cell_pp.dpkl')
        self.num_decoder_tokens, self.comments_pp = load_text_processor(os.getcwd() + '/data/comments_pp.dpkl')
        self.model_option = model_option.lower()

    def create_model(self):
        model_option = self.model_option
        # arbitrarly set latent dimension for embedding and hidden units
        latent_dim = 300

        ##### Define Model Architecture ######

        ########################
        #### Encoder Model ####
        encoder_inputs = Input(shape=(self.doc_length,), name='Encoder-Input')

        # Word embeding for encoder (ex: Cell)
        x = Embedding(self.num_encoder_tokens, latent_dim, name='Cell-Word-Embedding', mask_zero=False)(encoder_inputs)

        x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

        # We do not need the `encoder_output` just the hidden state.
        model_option = model_option.lower()
        if model_option == 'lstm':
            encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True, name='Encoder-Last-LSTM')(x)
            encoder_states = [state_h, state_c]
        elif model_option == 'bilstm':
            encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(
                LSTM(latent_dim, return_state=True), name='Bi-LSTM1')(x)
            state_h = Concatenate()([forward_h, backward_h])
            state_c = Concatenate()([forward_c, backward_c])
            encoder_states = [state_h, state_c]
            encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states, name='Encoder-Model')
        elif model_option == 'lstmattention':
            encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True, name='Encoder-Last-LSTM')(x)
            encoder_states = [state_h, state_c, encoder_outputs]
        else:
            encoder_outputs, state_h = GRU(latent_dim, return_state=True, name='Encoder-Last-GRU')(x)
            encoder_states = [state_h]

        # Encapsulate the encoder as a separate entity so we can just
        #  encode without decoding if we want to.
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states, name='Encoder-Model')

        seq2seq_encoder_out = encoder_model(encoder_inputs)

        ########################
        #### Decoder Model ####
        decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

        # Word Embedding For Decoder (ex: Commentss)
        dec_emb = Embedding(self.num_decoder_tokens, latent_dim, name='Decoder-Word-Embedding', mask_zero=False)(
            decoder_inputs)
        dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)
        # Set up the decoder, using `decoder_state_input` as initial state.
        if model_option == 'lstm':
            decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='Decoder-LSTM')
            output, _, _ = decoder_lstm(dec_bn, initial_state=seq2seq_encoder_out)
        elif model_option == 'lstmattention':
            o = Attention(name="Attention")(seq2seq_encoder_out)
            decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='Decoder-LSTM')
            # o = tf.repeat(o[:, tf.newaxis, :], 19, axis=1)
            output, _, _ = decoder_lstm(tf.concat([dec_bn, dec_bn * 0 + tf.expand_dims(o, axis=1)], axis=-1),
                                        initial_state=[state_h, state_c])
        elif model_option == 'bilstm':
            decoder_lstm = LSTM(latent_dim * 2, return_state=True, return_sequences=True, name='Decoder-LSTM')
            output, _, _ = decoder_lstm(dec_bn, initial_state=seq2seq_encoder_out)
        else:
            decoder_gru = GRU(latent_dim, return_state=True, return_sequences=True, name='Decoder-GRU')
            output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
        x = BatchNormalization(name='Decoder-Batchnorm-2')(output)

        # Dense layer for prediction
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='Final-Output-Dense')
        decoder_outputs = decoder_dense(x)
        ########################
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

    def train_model(self, batch_size=120, epochs=30):
        script_name_base = 'tutorial_seq2seq'
        csv_logger = CSVLogger(os.getcwd() + '/log/{:}.log'.format(script_name_base))
        model_checkpoint = ModelCheckpoint(
            os.getcwd() + '/checkpoint/{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
            save_best_only=True)
        history = self.model.fit([self.encoder_input_data, self.decoder_input_data],
                                 np.expand_dims(self.decoder_target_data, -1),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.12, callbacks=[csv_logger, model_checkpoint])
        self.model.save(os.getcwd()+'/' + self.model_option + '_seq2seq_model.h5')

    def predict_seq2seq_model(self, filename='/data/final_comments.csv'):
        seq2seq_model = tf.keras.models.load_model(os.getcwd()+'/' + self.model_option + '_seq2seq_model.h5')
        seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=self.cell_pp,
                                        decoder_preprocessor=self.comments_pp,
                                        seq2seq_model=seq2seq_model,
                                        model_option=self.model_option)
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
        start_time = time.time()
        predict_rows = pd.read_csv(os.getcwd() + '/data/predict_rows.csv')
        nums = predict_rows.shape[0]
        print("--- Predict %s ---" % self.model_option)
        pre = 0
        for i in range(nums):
            if isinstance(predict_rows.loc[i, 'conc_comment'], str):
              continue
            process = int(i * 100 / nums)
            if process % 10 == 0 and not pre == process:
                pre = process
                print('Complete process ' + str(process) + " percent in %s seconds" % (time.time() - start_time))
            _, predict_rows.loc[i, 'conc_comment'] = seq2seq_inf.generate_comments(
                predict_rows.loc[i, 'original_cell_no_comments'])       
        print("--- %s seconds ---" % (time.time() - start_time))
        comboo = predict_rows
        comboo = comboo.reset_index(drop=True)
        comboo['conc_comment'] = comboo['conc_comment'].apply(str)
        final_ans_no_false_prediction = comboo[comboo['conc_comment'].map(len) > 0]
        def get_cell_in_string(lst):
          sentence = ''
          for line in lst:
              sentence += line
          return sentence
        final_ans_no_false_prediction['conc_cell'] = final_ans_no_false_prediction['cells'].apply(get_cell_in_string)
        final_ans_no_false_prediction.to_csv(os.getcwd() + filename, index=False)

    def evaluate_seq2seq_model(self, nums=0):
        seq2seq_model = tf.keras.models.load_model(os.getcwd()+'/' + self.model_option + '_seq2seq_model.h5')
        seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=self.cell_pp,
                                        decoder_preprocessor=self.comments_pp,
                                        seq2seq_model=seq2seq_model,
                                        model_option=self.model_option)
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
        start_time = time.time()
        df_test_rows = pd.read_csv(os.getcwd() + '/data/df_test_rows.csv')
        nums = df_test_rows.shape[0] if nums == 0 else nums
        print("--- Test %s ---" % model_option)
        pre = 0
        for i in range(nums):
            process = int(i * 100 / nums)
            if process % 10 == 0 and not pre == process:
                pre = process
                print('Complete process ' + str(process) + " percent in %s seconds" % (time.time() - start_time))
            _, df_test_rows.loc[i, 'pred_comments'] = seq2seq_inf.generate_comments(
                df_test_rows.loc[i, 'original_cell_no_comments'])
        df_test_rows.to_csv(os.getcwd() + '/data/Seq2Seq_pred_comments.csv', index=False)
        print("--- %s seconds ---" % (time.time() - start_time))

        pred_csv = pd.read_csv(os.getcwd() + '/data/Seq2Seq_pred_comments.csv')
        pred_csv = pred_csv.dropna()
        original_com = pred_csv['conc_comment'].tolist()
        pred_com = pred_csv['pred_comments'].tolist()

        def remove_empty(a_str):
            if a_str == '':
                return False
            else:
                return True

        split_ori = []
        split_pre = []
        for i in range(min(len(original_com), nums)):
            ori = re.split(' |#|/|:|\.', original_com[i].lower())
            pre = re.split(' |#|/|:|\.', pred_com[i])
            ori_no_symbol = filter(remove_empty, ori)
            pre_no_symbol = filter(remove_empty, pre)

            split_ori.append(list(ori_no_symbol))
            split_pre.append(list(pre_no_symbol))

        scores1 = []
        scores2 = []
        scores3 = []
        scores4 = []
        scores_cummu = []
        for i in range(min(len(original_com), nums)):
            ori_sentence = [split_ori[i]]
            scores1.append(sentence_bleu(ori_sentence, split_pre[i], weights=(1, 0, 0, 0)))
            scores2.append(sentence_bleu(ori_sentence, split_pre[i], weights=(0, 1, 0, 0)))
            scores3.append(sentence_bleu(ori_sentence, split_pre[i], weights=(0, 0, 1, 0)))
            scores4.append(sentence_bleu(ori_sentence, split_pre[i], weights=(0, 0, 0, 1)))
            scores_cummu.append(sentence_bleu(ori_sentence, split_pre[i], weights=(0.15, 0.15, 0.35, 0.35)))
        mean_of_bleu1 = statistics.mean(scores1)
        mean_of_bleu2 = statistics.mean(scores2)
        mean_of_bleu3 = statistics.mean(scores3)
        mean_of_bleu4 = statistics.mean(scores4)
        mean_of_bleu_cummu = statistics.mean(scores_cummu)
        print('1-gram BLEU: ', mean_of_bleu1)
        print('2-gram BLEU: ', mean_of_bleu2)
        print('3-gram BLEU: ', mean_of_bleu3)
        print('4-gram BLEU: ', mean_of_bleu4)
        print('Cumulative BLUE: ', mean_of_bleu_cummu)
