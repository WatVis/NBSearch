from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for
import nmslib
import pandas as pd
import pickle as tastes_good
import gensim
import json
from flask import jsonify
from mult_alignment import mult_alignment
from scipy import spatial
import numpy as np
import ast
import io
import copy
import math
import logging
from numpy.linalg import norm



# load language model
the_lan_model = gensim.models.doc2vec.Doc2Vec.load('resource/my_model.doc2vec')
# load the dataframe
df_nb = pd.read_csv('resource/stored_df.csv')

# load the resource files generated from the backend
with open('resource/all_markdown_embeddings.txt', 'rb') as file:
    emb_vec_mkd = tastes_good.load(file)
with open('resource/all_code_embeddings.txt', 'rb') as file:
    emb_vec_code = tastes_good.load(file)
with open('resource/child_relationships.txt', 'rb') as file:
    relationship_arr = tastes_good.load(file)
with open('resource/dict.pkl', 'rb') as file:
    dict_relationships = tastes_good.load(file)


def cos_simi(x, emb):
    """
    A helper function which calculates cosine similarity
    -------------------
    Input:
    x => int or vector, int means we should ignore this cell
    emb => target vector, the one we should compare
    -------------------
    Output:
    Number from 0 to 2, easy to calculate rates in front-end
    """
    if type(x) == int:
        return 0
    new_cos_simi = 1 - spatial.distance.cosine(x, emb) + 1
    if (math.isnan(new_cos_simi)):
        print(x, emb)
    return new_cos_simi


def euc_dis(x, emb):
    """
    A helper function that calculates euclidean distance
    -------------------
    Input:
    x => int or vector, int means we should ignore this cell
    emb => target vector, the one we should compare
    -------------------
    Output:
    Number representing euclidean distance. 100 means we should
    ignore this distance.
    """
    if type(x) == int:
        return 100
    return norm(np.asarray(x) - np.asarray(emb))


def produce_list_json(df, neighbours):
    """
    Produce information needed by list_view
    -------------------
    Input:
    df => dataframe of cells in list_view
    neighbouts => int represents df length, aka number_of_search_results, or list_view length
    -------------------
    Output:
    a list of {'id': .., 'conc_cell': ...}
    id is notebook_id, conc_cell is the concatenation of all cell content
    """
    the_json = []
    for i in range(neighbours):
        curr_df = df[i]
        the_json.append({"id": str(curr_df['nb_id']), "conc_cell": curr_df['code_cell_no_comments']})

    return the_json


def produce_detail_json(list_of_df, neighbours, list_of_emb, one_emb):
    """
    Produces json needed by detail_view at the front end
    -------------------
    Input:
    list_of_df => a list of dataframes, each dataframe contains all cells from one notebook
    neighbours => int, length of list_of_df
    list_of_emb => 2D list contains all embeddings of cells in detail_view
    one_emb => the input query embedding
    -------------------
    Output:
    detail_json => 2D list of {'id': .., 'conc_cell': ...}, also contains markdown cells
    dist_json => 2D list of numbers, calculated by cosine similarity
    detail_json_no_markdown => 2D list of {'id': .., 'conc_cell': ...}
    dist_json_no_markdown => similar to dist_json, but does not consider markdown cells
    two_d_if_code => 2D list of numbers, each position represents a cell, 1 is code, 0 is markdown cell
    """
    detail_json = []
    detail_json_no_markdown = []
    two_d_if_code = []

    for i in range(neighbours):

        curr_df = list_of_df[i]

        curr_notebook = []
        curr_notebook_no_markdown = []
        one_d_if_code = []
        for index, row in curr_df.iterrows():
            if type(row['code_cell_no_comments']) != str:
                curr_notebook.append({"id": str(row['nb_id']), "conc_cell": row['markdown_cell'][2:][:-2]})
                one_d_if_code.append(0)
            else:
                curr_notebook.append({"id": str(row['nb_id']), "conc_cell": row['code_cell_no_comments']})
                curr_notebook_no_markdown.append({"id": str(row['nb_id']), "conc_cell": row['code_cell_no_comments']})
                one_d_if_code.append(1)

        two_d_if_code.append(one_d_if_code)
        detail_json.append(curr_notebook)
        detail_json_no_markdown.append(curr_notebook_no_markdown)

    # handle distance_json
    dist_json = []
    for i in range(neighbours):
        curr_notebook = list_of_emb[i]
        curr_cos_simi = []
        for j in range(len(curr_notebook)):
            curr_emb = curr_notebook[j]
            new_cos_simi = cos_simi(curr_emb, one_emb)
            curr_cos_simi.append(new_cos_simi)

        dist_json.append(curr_cos_simi)

    # handle another distance_json without markdown cells
    dist_json_no_markdown = []
    for i in range(neighbours):
        curr_df = list_of_df[i]
        curr_notebook_emb = list_of_emb[i]
        curr_cos_simi_no_markdown = []
        j = 0
        for index, row in curr_df.iterrows():
            if type(row['code_cell_no_comments']) == str:
                curr_emb = curr_notebook_emb[j]
                new_cos_simi = cos_simi(curr_emb, one_emb)
                curr_cos_simi_no_markdown.append(new_cos_simi)
            j = j + 1

        dist_json_no_markdown.append(curr_cos_simi_no_markdown)

    return detail_json, dist_json, dist_json_no_markdown, detail_json_no_markdown, two_d_if_code


def find_up_down(a_df, index, nbk_id):
    """
    Given a dataframe, find all rows with notebook_id == nbk_id
    -------------------
    Input:
    a_df => a dataframe contains all cells, cells with the same nbk_id should be adjacent
    index => index of target row
    nbk_id => notebook_id of the target row from a_df
    -------------------
    Output:
    sub-dataframe where all rows have notebook_id == nbk_id
    """
    up_most_index = find_up(a_df, index, nbk_id)
    down_most_index = find_down(a_df, index, nbk_id)

    up_index = up_most_index
    down_index = down_most_index

    return a_df.iloc[up_index:(down_index + 1)], emb_vec_code[up_index:(down_index + 1)]


def find_up(a_df, index, nbk_id):
    """
    find the earliest index with the same notebook id
    -------------------
    Input:
    a_df => a dataframe contains all cells, cells with the same nbk_id should be adjacent
    index => index of target row
    nbk_id => notebook_id of the target row from a_df
    -------------------
    Output:
    An index, which represents the smallest index with notebook_id == nbk_id
    """
    # check if it's the first
    if index == 0:
        return index
    # sure that it starts from at least 1
    while index > 0:
        if (a_df.iloc[index - 1].nb_id == nbk_id):
            index = index - 1
        else:
            return index
    # check if index 0 is in the range
    # index must be equal to 0 now, o/w it will be returned in the while loop
    if (a_df.iloc[index].nb_id == nbk_id):
        return index
    # seems like the first row is from another notebook, different from row 2
    return index + 1


def find_down(a_df, index, nbk_id):
    """
    find the latest index within the same notebook
    -------------------
    Input:
    a_df => a dataframe contains all cells, cells with the same nbk_id should be adjacent
    index => index of target row
    nbk_id => notebook_id of the target row from a_df
    -------------------
    Output:
    An index, which represents the largest index with notebook_id == nbk_id
    """
    # check if it's the last
    if index == (a_df.shape[0] - 1):
        return index

    # sure that there are rows undernth
    while index < (a_df.shape[0] - 1):
        if (a_df.iloc[index + 1].nb_id == nbk_id):
            index = index + 1
        else:
            return index
    # check if the last row is in the range
    # index must be equal to a_df.shape[0] - 1
    if (a_df.iloc[index].nb_id == nbk_id):
        return index
    # the last row is not the correct answer, go back
    return index - 1

def skip_markdown(lt):
    """
    used after Nov 15. we have different info in alignment and detail views
    we use this function to skip markdown cells
    -------------------
    Input:
    lt => 2D list of ints, 1 is code, 0 is markdown cell, each list represents one notebook
    -------------------
    Output:
    2D list of ints, each list contains index of code cells.
    """
    skip_notebooks = []
    for notebook in lt:
        skip_a_notebook = []
        for j in range(len(notebook)):
            if notebook[j] == 1:
                skip_a_notebook.append(j)
        skip_notebooks.append(skip_a_notebook)
    return skip_notebooks


################# Can ignore this function since it's related to Dots_view #######################
def alignment_them(two_d_arr):
    """
    THIS FUNCTION IS FOR DOTS_VIEW. CHECK 'unaligned_result(...)' FOR LINES_VIEW.
    AFTER RUNNING alignment_them(), YOU SHOULD ALSO RUN process_path() TO GET ALL PATH &
    NODES INFORMATION OF DOTS_VIEW, BUT YOU ONLY NEED ONE FUNCTION 'unalign_result(...)'
    FOR LINES_VIEW. CHECK 'search1(...)' FOR HOW TO USE THESE FUNCTIONS.


    Generates information needed by alignment_view, specifically Dots_view. MSA is applied here.
    The alignment process is ignoring markdown cells.
    -------------------
    Input:
    two_d_arr => 2D array of embeddings, each array is representing an entire notebook.
    -------------------
    Output:
    2D array of ints, each array represents a column in Dots_view, e.g.
    if the FIRST array is [2,1,2,2,2,1], that means, in Dots_view, the first position of column 1
    is occupied by the first cell of notebook 2, and notebook 2 has at least 4 cells(because we
    can find 4 cells pushed to the left to occupy notebook 1's empty positions).
    """
    align_res = mult_alignment(two_d_arr)

    # make all - to 0, o/w 1
    bit_arr = []
    for i in range(len(align_res)):
        an_bit = []
        for j in range(len(align_res[i])):
            if (align_res[i][j] == -100):
                an_bit.append(0)
            else:
                an_bit.append(i + 1)
        bit_arr.append(an_bit)
    # push them to the left!
    for i in range(len(bit_arr)):
        # if i == 0, then skip this
        if i == 0:
            # do nothing
            continue
        else:
            for j in range(len(bit_arr[i])):
                # find the left-most position j can go
                if bit_arr[i][j] == 0:
                    continue
                else:
                    curr_left_most = i
                    for k in reversed(list(range(i))):
                        if bit_arr[k][j] == 0:

                            curr_left_most = k
                        else:
                            # by promise, the previous should be fine
                            break

                    if curr_left_most == i:
                        # no shifting happend
                        continue
                    else:
                        # 1. change curr_left_most position to bit_arr[i][j]
                        bit_arr[curr_left_most][j] = bit_arr[i][j]
                        # 2. change to 0 all the way from curr_left + 1 to i
                        for m in list(range(curr_left_most, i)):
                            bit_arr[m + 1][j] = 0


    # now the bit_arr stores the shifted version
    # based on the shifted version, we need to know, where to find my (current line) bits
    position_arr = []
    for i in range(len(bit_arr)):
        # list of zeros
        position_arr.append([0] * len(bit_arr[0]))
    for i in range(len(bit_arr)):
        # index 0 with cell content should be 1, and so on. we use
        # 0 to indicate 'nothing'.
        curr_number_should_be = i + 1
        for j in range(len(bit_arr[i])):
            if bit_arr[i][j] != 0:
                # indicate 'it is in my place'
                position_arr[bit_arr[i][j] - 1][j] = curr_number_should_be

    return position_arr


def create_markdown_path(idx, start, mid):
    """
    visualize markdown cells in alignment_view
    -------------------
    Input:
    idx => a 2-d list of ints, where 0 represents markdown and 1 represents a code cell
    start => 1D list of [num(x position), num(y position)], represents first cells from all notebooks
    mid => 2D list of [num(x position), num(y position)], all cells except first ones from all notebooks
    -------------------
    Output:
    2D array of {'y': ..., 'left': ..., 'right': ...}, these information is needed to render markdown cells
    in alignment_view
    """
    y_shift = 10
    x_shift = 20
    interval_x = 20
    interval_y = 15
    width = 10

    markdown_height = 4

    markdown_path = []
    for i in range(len(idx)):
        single_path = []
        for j in range(len(idx[i])):


            if j == 0:
                if idx[i][j] > 0:
                    single_path.append([{'y': start[i][1] / 2 - markdown_height / 2,
                                         'left': x_shift + i * interval_x,
                                         'right': x_shift + i * interval_x + width},
                                        {'y': start[i][1] / 2 + markdown_height / 2,
                                         'left': x_shift + i * interval_x,
                                         'right': x_shift + i * interval_x + width}])
            elif j == 1:
                if idx[i][j] - idx[i][j - 1] > 1:
                    single_path.append([{'y': (start[i][1] + mid[i][j - 1][1]) / 2 - markdown_height / 2,
                                         'left': x_shift + i * interval_x,
                                         'right': x_shift + i * interval_x + width},
                                        {'y': (start[i][1] + mid[i][j - 1][1]) / 2 + markdown_height / 2,
                                         'left': x_shift + i * interval_x,
                                         'right': x_shift + i * interval_x + width}])
            elif idx[i][j] - idx[i][j - 1] > 1:
                single_path.append([{'y': (mid[i][j - 2][1] + mid[i][j - 1][1]) / 2 - markdown_height / 2,
                                     'left': x_shift + i * interval_x,
                                     'right': x_shift + i * interval_x + width},
                                    {'y': (mid[i][j - 2][1] + mid[i][j - 1][1]) / 2 + markdown_height / 2,
                                     'left': x_shift + i * interval_x,
                                     'right': x_shift + i * interval_x + width}])
        markdown_path.append(single_path)
    return markdown_path



def process_path(alignment):
    """
    USED AFTER RUNNING 'alignment_them' AS A COMBINATION.

    example ====>>>>
    input data format:  var alignment = [[1,0,0,1,1,0],
                                         [2,1,0,2,2,1]]
    output data format: var paths = [[{y:0, left: 30, right: 40}],
                                     [{y:0, left: 30, right: 40}]]
                        var start_pts = [[x_value, y_value], ...]
                        var middle_pts = [[[x_value, y_value], ...],
                                          [...]]
    """
    # shifting based on the coordinate origin
    y_shift = 10
    x_shift = 20
    interval_x = 20
    interval_y = 15
    width = 10

    path_result = []
    start_pts = []
    middle_pts = []

    for i in range(len(alignment)):
        single_path = []
        first = True
        single_middle = []
        for j in range(len(alignment[i])):
            if alignment[i][j] == 0:
                single_path.append(None)
            else:
                single_path.append({'y': y_shift + j * interval_y,
                                    'left': x_shift + (alignment[i][j] - 1) * interval_x,
                                    'right': x_shift + (alignment[i][j] - 1) * interval_x + width})
                if first:
                    first = False
                    start_pts.append([x_shift + (alignment[i][j] - 1) * interval_x + width / 2, y_shift + j * interval_y])
                else:
                    single_middle.append([x_shift + (alignment[i][j] - 1) * interval_x + width / 2, y_shift + j * interval_y])
        path_result.append(single_path)
        middle_pts.append(single_middle)
    return path_result, start_pts, middle_pts


def unaligned_result(two_d_arr):
    """
    THIS FUNCTION IS FOR LINES_VIEW. CHECK 'alignment_them(...)' FOR DOTS_VIEW.
    AFTER RUNNING 'alignment_them()'', YOU SHOULD ALSO RUN process_path() TO GET ALL PATH &
    NODES INFORMATION OF DOTS_VIEW, BUT YOU ONLY NEED ONE FUNCTION 'unalign_result(...)'
    FOR LINES_VIEW. CHECK 'search1(...)' FOR HOW TO USE THESE FUNCTIONS.


    Generates information needed by alignment_view, specifically Dots_view. MSA is applied here.
    The alignment process is ignoring markdown cells.
    -------------------
    Input:
    two_d_arr => 2D array of embeddings, each array is representing an entire notebook.
    -------------------
    Output:
    2D array of ints, each array represents a column in Dots_view, e.g.
    if the FIRST array is [2,1,2,2,2,1], that means, in Dots_view, the first position of column 1
    is occupied by the first cell of notebook 2, and notebook 2 has at least 4 cells(because we
    can find 4 cells pushed to the left to occupy notebook 1's empty positions).
    """
    align_res = mult_alignment(two_d_arr)

    # make all - to 0, o/w 1
    bit_arr = []
    for i in range(len(align_res)):
        an_bit = []
        for j in range(len(align_res[i])):
            if (align_res[i][j] == -100):
                an_bit.append(0)
            else:
                an_bit.append(i + 1)
        bit_arr.append(an_bit)

    # shifting based on the coordinate origin
    y_shift = 10
    x_shift = 20
    interval_x = 20
    interval_y = 15
    width = 10

    path_result = []
    start_pts = []
    middle_pts = []

    for i in range(len(bit_arr)):
        # find ending position & starting position index
        first = 0
        while bit_arr[i][first] == 0:
            first = first + 1
        end = len(bit_arr[i]) - 1
        while bit_arr[i][end] == 0:
            end = end - 1
        ########### found start & end! ##################

        # deal with path
        single_path = []
        # deal with middle
        single_middle = []
        # deal with start
        start_pts.append([x_shift + (bit_arr[i][first] - 1) * interval_x + width / 2, y_shift + first * interval_y])

        for j in range(len(bit_arr[i])):
            if j < first or j > end:
                single_path.append(None)
            else:
                single_path.append({'y': y_shift + j * interval_y,
                                    'left': x_shift + i * interval_x,
                                    'right': x_shift + i * interval_x + width})

                if bit_arr[i][j] != 0 and j != first:
                    single_middle.append(
                        [x_shift + (bit_arr[i][j] - 1) * interval_x + width / 2, y_shift + j * interval_y])

        path_result.append(single_path)
        middle_pts.append(single_middle)
    return path_result, start_pts, middle_pts



##################### This section deal with linking problem ####################
def most_frequent(List):
    """
    Find the most frequent element from an array/list
    Could implement by a hashtable to improve efficiency
    -------------------
    Input:
    List => a list of something
    -------------------
    Output:
    the element with the most duplicates.
    """
    if len(List) == 0:
        return None
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


def generate_relationship_array(detail_view):
    """
    We compute arrow relationships here.
    -------------------
    Input:
    detail_view => 2D array of {'id': xxx, 'conc_cell': xxx}, contains all cells
    -------------------
    Output:
    store_all_variables => 2D array of (array of strings),
                            each string is an variable name that cell has

    all_relationships_with_name => 2D array of (array of {'index': xxx, 'var_name': xxx}),
                            each 'index' is another cell from the same notebook,
                            each 'var_name' is the variable name that shared between me and
                                    the cell with that 'index'

    all_relationships => 2D array of (array of int), each int has the same meaning as above ('index')
                        e.g. [[[2,3,3,3], ...], ...] means cell 1 of notebook 1 has 1 shared variable with
                        cell 2 of notebook 1, and 3 shared variables with cell 3 of notebook 1.
    """
    # corpus: all notebooks.
    all_relationships = []
    all_relationships_with_name = []
    store_all_variables = []

    for i in range(len(detail_view)):
        # first, let's see variable names all cells have
        var_names = []
        for j in range(len(detail_view[i])):
            # variable names from current cell
            curr_cell_var_name = []
            if type(detail_view[i][j]['conc_cell']) != str:
                var_names.append([])
                continue
            buffer = io.StringIO(detail_view[i][j]['conc_cell'])
            line = buffer.readline()
            while line:
                try:
                    root = ast.parse(line)
                    names = sorted({node.id for node in ast.walk(root) if isinstance(node, ast.Name)})
                    for var_name in names:
                        if var_name in curr_cell_var_name:
                            # do nothing
                            nothing = 0
                        else:
                            curr_cell_var_name.append(var_name)
                except:
                    # do nothing
                    nothing = 0
                line = buffer.readline()
            var_names.append(curr_cell_var_name)
        store_all_variables.append(var_names)

        # deal with all_relationships
        one_nb_all_relationship = []
        one_nb_all_relationship_with_name = []

        for j in range(len(var_names)):
            # cell #j's relationship
            store_rel = []
            store_rel_with_name = []
            for one_var_name in var_names[j]:
                # record all cells containg this variable
                for k in range(len(var_names)):
                    if j == k:
                        continue
                    else:
                        if one_var_name in var_names[k]:
                            store_rel.append(k)
                            store_rel_with_name.append({'index': k, 'var_name': one_var_name})
            one_nb_all_relationship.append(store_rel)
            one_nb_all_relationship_with_name.append(store_rel_with_name)
        all_relationships.append(one_nb_all_relationship)
        all_relationships_with_name.append(one_nb_all_relationship_with_name)

    # note that all_relationships has duplicates, because we did not check uniqueness in
    #    line 50. But this is a good thing, cuz we can do voting at the end.
    return all_relationships, all_relationships_with_name, store_all_variables


def counter(names):
    """
    Example:
    converts [1,2,2,2,3,3] into {1:1, 2:3, 3:2}
    -------------------
    Input:
    names => array
    -------------------
    Output:
    a dictionary
    """
    d = {}
    for name in names:
        d[name] = d.get(name, 0) + 1
    return d



def simplify_array(all_relationships, all_relationships_with_name):
    """
    Used after generate_relationship_array(...).
    Extract information from the input arrays, and produce a more informatic array.
    -------------------
    Input:
    all_relationships => check generate_relationship_array(...)'s output
    all_relationships_with_name => check generate_relationship_array(...)'s output
    -------------------
    Output:
    2D array of (array of {'index': xxx, 'count': xxx, 'position': [], 'var_names': xxx})
    we set 'position' as [] because we will contribute that part in later functions
    check generate_unshift_arrow(...) and generate_shift_arrow(...)
    """
    all_relationships_dict = []
    for i in range(len(all_relationships)):
        nb_dict = []
        for j in range(len(all_relationships[i])):
            cell_rel = all_relationships[i][j]
            cell_dict = []
            # cell_count_dict is {1:1, 2:1, 3:5}
            cell_count_dict = counter(cell_rel)
            for key in cell_count_dict:
                var_names = []
                for obj in all_relationships_with_name[i][j]:
                    if obj['index'] == key:
                        var_names.append(obj['var_name'])
                cell_dict.append({'index': key, 'count': cell_count_dict[key], 'position': [], 'var_names': var_names})
            nb_dict.append(cell_dict)
        all_relationships_dict.append(nb_dict)

    return all_relationships_dict

def generate_unshift_arrow(rel, starter, middle):
    """
    This function is being used after simplify_array(...) for filling
                element 'position' with [start, middle, end]
    when you click a cell in alignment_view, it shows arrows.
    The arrow positions are actually generated here, in backend.
    -------------------
    Input:
    rel => check output of simplify_array(...)
    starter => check output of unaligned_result(...)
    middle => check output of unaligned_result(...)
    -------------------
    Output:
    similar with input 'rel', but fills 'position' with [start, middle, end]
    start represents starting position of the arrow, middle is the middle point,
    since we'd like the arrow to have some curvity. 'end' is where the arrow points at.
    each of the 'start', 'middle', and 'end' is in form of [x, y], i.e. list of 2 numbers.
    """
    interval_x = 20
    for i in range(len(rel)):
        for j in range(len(rel[i])):
            for k in range(len(rel[i][j])):

                if rel[i][j][k]['index'] < j:
                    # destination is higher
                    if rel[i][j][k]['index'] == 0:
                        rel[i][j][k]['position'] = [[middle[i][j - 1],
                                                     [middle[i][j - 1][0] + interval_x / 2, (middle[i][j - 1][1] + starter[i][1]) / 2],
                                                     starter[i]]]
                    else:
                        rel[i][j][k]['position'] = [[middle[i][j - 1],
                                                     [middle[i][j - 1][0] + interval_x / 2,
                                                      (middle[i][j - 1][1] + middle[i][rel[i][j][k]['index'] - 1][1]) / 2],
                                                     middle[i][rel[i][j][k]['index'] - 1]]]
                else:
                    # destination is lower than me
                    if j == 0:
                        rel[i][j][k]['position'] = [[starter[i],
                                                     [middle[i][rel[i][j][k]['index'] - 1][0] + interval_x / 2,
                                                      (middle[i][rel[i][j][k]['index'] - 1][1] + starter[i][1]) / 2],
                                                     middle[i][rel[i][j][k]['index'] - 1]]]
                    else:
                        rel[i][j][k]['position'] = [[middle[i][rel[i][j][k]['index'] - 1],
                                                     [middle[i][j - 1][0] + interval_x / 2,
                                                      (middle[i][j - 1][1] + middle[i][rel[i][j][k]['index'] - 1][1]) / 2],
                                                     middle[i][j - 1]]]

    return rel


def generate_shift_arrow(rel, starter, middle):
    """
    This function is being used after process_path(...) for filling
                element 'position' with [start, middle, end].
                More specifically, for DOTS_VIEW.
    when you click a cell in alignment_view, it shows arrows.
    The arrow positions are actually generated here, in backend.
    -------------------
    Input:
    rel => check output of simplify_array(...)
    starter => check output of unaligned_result(...)
    middle => check output of unaligned_result(...)
    -------------------
    Output:
    similar with input 'rel', but fills 'position' with [start, middle, end]
    start represents starting position of the arrow, middle is the middle point,
    since we'd like the arrow to have some curvity. 'end' is where the arrow points at.
    each of the 'start', 'middle', and 'end' is in form of [x, y], i.e. list of 2 numbers.
    """
    interval_y = 15
    interval_x = 20


    for i in range(len(rel)):
        for j in range(len(rel[i])):
            for k in range(len(rel[i][j])):

                if rel[i][j][k]['index'] < j:
                    # destination is higher
                    if rel[i][j][k]['index'] == 0:
                        rel[i][j][k]['position'] = [[middle[i][j - 1],
                                                     [(middle[i][j - 1][0] + starter[i][0]) / 2 + interval_x / 2,
                                                      (middle[i][j - 1][1] + starter[i][1]) / 2],
                                                     starter[i]]]
                    else:
                        rel[i][j][k]['position'] = [[middle[i][j - 1],
                                                     [(middle[i][j - 1][0] + middle[i][rel[i][j][k]['index'] - 1][0]) / 2 + interval_x / 2,
                                                      (middle[i][j - 1][1] + middle[i][rel[i][j][k]['index'] - 1][1]) / 2 + interval_y / 2],
                                                     middle[i][rel[i][j][k]['index'] - 1]]]
                else:
                    # destination is lower than me
                    if j == 0:
                        rel[i][j][k]['position'] = [[starter[i],
                                                     [(middle[i][rel[i][j][k]['index'] - 1][0] + starter[i][0]) / 2 + interval_x / 2,
                                                      (middle[i][rel[i][j][k]['index'] - 1][1] + starter[i][1]) / 2],
                                                     middle[i][rel[i][j][k]['index'] - 1]]]
                    else:
                        rel[i][j][k]['position'] = [[middle[i][rel[i][j][k]['index'] - 1],
                                                     [(middle[i][j - 1][0] + middle[i][rel[i][j][k]['index'] - 1][0]) / 2 + interval_x / 2,
                                                      (middle[i][j - 1][1] + middle[i][rel[i][j][k]['index'] - 1][1]) / 2 + interval_y / 2],
                                                     middle[i][j - 1]]]

    return rel
##################################################################################


def search1(query, neighbours=5, alpha=0.42, beta=0.58):
    """
    This function receives search query and produces required information
    for further analysis
    -------------------
    Input:
    query => the input query received from user
    neighbours => default is 5, number of notebooks we'd like to see
    alpha, beta => markdown impact factor, alpha + beta = 1,
                    beta is the importance of markdown cells.
    -------------------
    Output:
    check produce_list_json(...) and produce_detail_json(...) for details.
    """
    new_emb = the_lan_model.infer_vector(gensim.utils.simple_preprocess(query))
    # None will be 0, so that argmax will not take them.
    # The decay process will also have 0 impact.
    dist_mkd = [euc_dis(x, new_emb) for x in emb_vec_mkd]
    dist_code = [euc_dis(x, new_emb) for x in emb_vec_code]

    # apply dist_mkd to dist_code:
    for dic in dict_relationships:
        child_code_arr = dict_relationships[str(dic)]
        length = len(child_code_arr)
        for i in range(length):
            decay = (length - i) / length
            dist_code[child_code_arr[i]] = alpha * dist_code[child_code_arr[i]]
            + beta * dist_mkd[int(dic)]
    # take argmin
    #     sort_them = np.argsort(dist_code)[::-1][:k]
    sort_them = np.argsort(dist_code)[:neighbours]
    # this is for the list_view
    store_rows = []
    # these 2 are for the detailed view
    store_df = []
    all_embeddings = []

    for idx in sort_them:
        # create the list
        cells = df_nb.iloc[idx]
        store_rows.append(cells)
        # create the detailed view
        notebook_id = df_nb.iloc[idx].nb_id
        new_dataframe, new_embeddings = find_up_down(df_nb, idx, notebook_id)
        store_df.append(new_dataframe)
        all_embeddings.append(new_embeddings)

    detail_json, emb_json, emb_json_no_markdown, detail_json_no_markdown, two_d_if_code = produce_detail_json(store_df, neighbours, all_embeddings, new_emb)
    return produce_list_json(store_rows, neighbours), detail_json, emb_json, emb_json_no_markdown, detail_json_no_markdown, skip_markdown(two_d_if_code)



app = Flask(__name__)

@app.route('/data/<path:path>')
def send_json(path):
    print("sent?")
    return send_from_directory('data', path)



@app.route('/', methods=['POST', 'GET'])
def find_query():
    print(request.method)
    if request.method == 'POST':
        sentences = request.form["search_query"]
        nbk_search_rslts = request.form["nbk_results_count"]
        mkd_sig = request.form["mkd_significance"]
        beta_rev = float(mkd_sig)
        alpha_rev = 1 - beta_rev

        list_view = []
        detail_view = []
        emb_json = []

        # list_view: a list of {id: ..., conc_cells}, check produce_list_json
        # detail_view: a 2-d list, similar to list_view, but detail_view contains all information (all cells from every notebook)
        # emb_json: 2-d list contains all embedding information, emb_json_no_markdown does not have embedding for markdown cells
        # detail_json_no_markdown is detail_view without markdown cells
        # real_index is a 2-d list, where 0 represents markdown and 1 represents a code cell
        list_view, detail_view, emb_json, emb_json_no_markdown, detail_json_no_markdown, real_index = search1(sentences, neighbours=int(nbk_search_rslts), alpha=alpha_rev, beta=beta_rev)

        # this will give you aligned paths, starters & middle cells
        alignment_position_result = alignment_them(emb_json_no_markdown)

        # these are information needed by D3 library to draw paths and nodes
        path_position_result, starter, middle = process_path(alignment_position_result)

        # this will give you unaligned paths, starters & middle cells
        unaligned_path, unaligned_starter, unaligned_middle = unaligned_result(emb_json_no_markdown)

        # deal with arrows here    ##########################################################################
        rel, rel_with_name, all_variables = generate_relationship_array(detail_json_no_markdown)

        simple_rel = simplify_array(rel, rel_with_name)

        simple_rel_copy = copy.deepcopy(simple_rel)

        unshifted_arrow = generate_unshift_arrow(simple_rel, unaligned_starter, unaligned_middle)

        shifted_arrow = generate_shift_arrow(simple_rel_copy, starter, middle)

        useless1, useless2, all_variables = generate_relationship_array(detail_view)

        #####################################################################################################
        # markdown cells are shown in alignment_view, this is the information needed to render markdown cells.
        # markdown_path_data = create_markdown_path(real_index, starter, middle)
        markdown_path_data = create_markdown_path(real_index, unaligned_starter, unaligned_middle)
        #####################################################################################################





        return render_template('search.html', data=list_view, details=detail_view, emb=emb_json,
                               align_paths=path_position_result, align_start=starter, align_mid=middle,
                               unalign_paths=unaligned_path, unalign_start=unaligned_starter,
                               unalign_mid=unaligned_middle,
                               unshift_arrow=unshifted_arrow, shift_arrow=shifted_arrow,
                               the_query=sentences, all_variables=all_variables,
                               returned_variables=nbk_search_rslts, actual_index=real_index,
                               mkd_path=markdown_path_data, prev_beta=beta_rev*100,
                               show_it=1)
    return render_template('search.html', title='Home', show_it=0)




if __name__ == '__main__':
    app.run(debug=True)
