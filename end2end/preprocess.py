import ast
import dill as dpickle
import io
import json
import numpy as np
import os
import pandas as pd
import spacy
import statistics
import tensorflow as tf
import tokenize

from ktext.preprocess import processor
from nltk.tokenize import RegexpTokenizer
from os import listdir, mkdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

def preprocessing(path):
    """
    Preprocess notebooks data.
    Parameters
    ----------
    path: str
        the path of the directory for notebooks data
    -------
    Typical Usage:
    -------------
    preprocessing('data/notebooks/')
    """
    print("Start preprocessing text")
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = {'nb_id': files}
    df_nb = pd.DataFrame(data, columns=['nb_id'])
    dirName = 'checkpoint'
    try:
        # Create target Directory
        mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    dirName = 'log'
    try:
        # Create target Directory
        mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    def get_num_cells(nb_id):
        nb_name = path + nb_id

        with open(nb_name) as nb_file:

            try:
                # get the nb as a JSON file
                data = json.load(nb_file)
                if isinstance(data, dict):
                    keys = data.keys()
                else:
                    keys = []

                # get the number of cells
                if 'cells' in keys:
                    return len(data['cells'])
                elif 'worksheets' in keys:
                    num_cells = 0
                    for w in data['worksheets']:
                        num_cells += len(w['cells'])
                    return num_cells

            except:
                return None

    df_nb['num_cells'] = df_nb['nb_id'].apply(get_num_cells)
    df_nb = df_nb.query("num_cells > 0")
    df_nb = df_nb.reset_index(drop=True)

    def keep_code(something):
        if something.get('cell_type') == "code":
            return True
        return False

    def keep_source_code(something):
        if something.get('source') == None:
            return something.get('input')
        return something.get('source')

    def get_codes_from_name(nb_id):
        nb_name = path + nb_id

        with open(nb_name) as nb_file:

            try:
                # get the nb as a JSON file
                data = json.load(nb_file)
                if isinstance(data, dict):
                    keys = data.keys()
                else:
                    keys = []

                # get the number of cells
                if 'cells' in keys:
                    iterable = data['cells']
                    itor = list(filter(keep_code, iterable))
                    itor = list(map(keep_source_code, itor))
                    return itor
                elif 'worksheets' in keys:
                    cells = []
                    for w in data['worksheets']:
                        cells.append(w['cells'])
                    flattened_list = [y for x in cells for y in x]
                    itor = list(filter(keep_code, flattened_list))
                    itor = list(map(keep_source_code, itor))
                    return itor

            except:
                return None

    df_nb['cells'] = df_nb['nb_id'].apply(get_codes_from_name)
    df_nb = df_nb.drop(columns=['num_cells'])
    df_nb = df_nb.explode('cells').reset_index(drop=True)

    # try to extract comments
    # Lots of people submitted code that has syntax error......................
    # i'll just remove those. These should not be legal code

    # also, stuff like this: #for variable in field.getchildren():
    # should not be considered as useful comments. Since it is not natural language

    # just a way to detect if we run into a scenario mentioned above
    def is_valid_python(code):
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True

    def get_comments(lst):
        try:
            if lst == []:
                return []
            the_whole_cell = ''
            for li in lst:
                the_whole_cell += li

            buf = io.StringIO(the_whole_cell)
            ans = []
            for line in tokenize.generate_tokens(buf.readline):
                if line.type == tokenize.COMMENT:

                    # check if you have things like "#for variable in field.getchildren():"
                    if (is_valid_python(line.string.strip("#").strip(" ").strip("#"))):
                        continue
                    else:
                        ans.append(line.string)
            return ans
        except:
            # your code has syntax errors....................
            return "Syntax_error,srsly?"

    df_nb['comments'] = df_nb['cells'].apply(get_comments)
    # drop those rows have "Syntax_error,srsly?"
    df_nb = df_nb[df_nb.comments != 'Syntax_error,srsly?']
    df_nb = df_nb.reset_index(drop=True)

    # get pure code
    def remove_comments(lst):
        if lst == []:
            return ''
        the_whole_cell = ''
        for li in lst:
            the_whole_cell += li

        buf = io.StringIO(the_whole_cell)
        ans = ''
        for line in tokenize.generate_tokens(buf.readline):
            if line.type != tokenize.COMMENT:
                ans += line.string + ' '
        return ans

    df_nb['original_cell_no_comments'] = df_nb['cells'].apply(remove_comments)

    def isEnglish(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def remove_non_ascii_comments(lst):
        ans = []
        for cmt in lst:
            if (isEnglish(cmt)):
                ans.append(cmt)
            else:
                continue
        return ans

    df_nb['comments'] = df_nb['comments'].apply(remove_non_ascii_comments)

    # concatenate all comments
    def concatenate_valid_comments(lst):
        if lst == []:
            return ''
        ans = ''
        for cmt in lst:
            ans += cmt + ' '
        return ans

    df_nb['conc_comment'] = df_nb['comments'].apply(concatenate_valid_comments)

    EN = spacy.load('en_core_web_sm')

    def tokenize_code(text):
        # tokenize code strings
        return RegexpTokenizer(r'\w+').tokenize(text)

    df_nb['function_token'] = df_nb['original_cell_no_comments'].apply(tokenize_code)

    def tokenize_comments(lst):
        if lst == []:
            return []
        the_whole_cmt = ''
        for cmt in lst:
            the_whole_cmt += ' ' + cmt
        return tokenize_code(the_whole_cmt)

    df_nb['comment_token'] = df_nb['comments'].apply(tokenize_comments)
    df_nb_non_empty_function_body = df_nb[df_nb['function_token'].map(len) > 0]
    df_train_rows = df_nb_non_empty_function_body[
        df_nb_non_empty_function_body['comment_token'].map(len) > 0].reset_index(drop=True)
    df_predict_rows = df_nb_non_empty_function_body
    train_func_raw = df_train_rows.original_cell_no_comments.tolist()
    train_cmts_raw = df_train_rows.conc_comment.tolist()
    df_train_rows.to_csv(os.getcwd() + '/data/train_rows.csv')
    df_predict_rows.to_csv(os.getcwd() + '/data/predict_rows.csv')
    df_train_rows, df_test_rows = train_test_split(df_train_rows, test_size=.10)

    df_test_rows.to_csv(os.getcwd() + '/data/df_test_rows.csv')
    # preview output of first element
    func_pp = processor(keep_n=8000, padding_maxlen=100)
    train_func_vecs = func_pp.fit_transform(train_func_raw)
    comments_pp = processor(append_indicators=True, keep_n=4500,
                            padding_maxlen=20, padding='post')
    train_comments_vecs = comments_pp.fit_transform(train_cmts_raw)
    # Save preprocessors
    with open(os.getcwd() + '/data/cell_pp.dpkl', 'wb') as f:
        dpickle.dump(func_pp, f)
    with open(os.getcwd() + '/data/comments_pp.dpkl', 'wb') as f:
        dpickle.dump(comments_pp, f)

    # Save the processed data
    np.save(os.getcwd() + '/data/train_comments_vecs.npy', train_comments_vecs)
    np.save(os.getcwd() + '/data/train_cell_vecs.npy', train_func_vecs)
    print("Finish processing data")

def preprocess_language_model_data():
    """
    Preprocess notebooks data for doc2vec model.
    """
    predicted = pd.read_csv(os.getcwd() + '/data/train_rows.csv')
    f= open(os.getcwd() + '/data/corpus.txt',"w+")
    for index, row in predicted.iterrows():
        f.write(row['conc_comment'] + '\n')
    f.close()