# This is a program that aligns cells from multiple notebooks
#   similarity between two cells can be calculated using a function passed as parameter
import timeit
import heapq
import sys
from random import randrange

# functions for debug purpose
DEBUG = False


def debug_gen_similarity_map():
    debug_cell_types = ['a', 'b', 'c', 'd', 'e', 'f']
    main_map = {}
    for c in debug_cell_types:
        curr_map = {}
        for k in debug_cell_types:
            curr_map[k] = randrange(10)
        main_map[c] = curr_map
    for c in debug_cell_types:
        for k in debug_cell_types:
            main_map[c][k] = main_map[k][c]
    return main_map


debug_map = debug_gen_similarity_map()


# if DEBUG:
#     print(debug_map)

def debug_similarity(cell1, cell2):
    return debug_map[cell1][cell2]


def similartiy_func1(cell1, cell2):
    return 1 / (1 + abs(ord(cell1) - ord(cell2)))


def similarity_func2(cell1, cell2):

    # ans = 1 / (1 + abs(float(cell1) - float(cell2)))
    ans = abs(float(cell1) - float(cell2))
    return ans


# Helper for pairwise alignment
# find the optimal j value where we divide seq2
#   seq1 and seq2 are lists of cell classes
def find_optimal(seq1, seq2, cell_similarity):
    indel = 0  # self-defined

    len1 = len(seq1)
    len2 = len(seq2)
    if len1 < 2 or len2 < 2:  # not suppose to happen
        Exception

    # construct dp tables with 2 rows
    front_tbl = []
    front_tbl.append([])
    front_tbl.append([])
    for j in range(0, len2 + 1):
        front_tbl[0].append(indel * j)  # setting the initial first row
        front_tbl[1].append(indel)
    reverse_tbl = []
    reverse_tbl.append([])
    reverse_tbl.append([])
    for j in range(0, len2 + 1):
        reverse_tbl[0].append(indel * j)  # setting the initial first row
        reverse_tbl[1].append(indel)

    # compute j which has the maximum alignment score between seq1[0, len1/2] and seq2[0, j]
    for i in range(0, len1 // 2):
        front_tbl[1][0] = indel * (i + 1)  # setting the first column
        for j in range(1, len2 + 1):
            cell1 = seq1[i]
            cell2 = seq2[j - 1]
            similarity = cell_similarity(cell1, cell2)
            score1 = front_tbl[0][j - 1] + similarity
            score2 = front_tbl[0][j] + indel
            score3 = front_tbl[1][j - 1] + indel
            curr_score = max(score1, score2, score3)
            front_tbl[1][j] = curr_score
        # reset the table
        for j in range(0, len2 + 1):
            front_tbl[0][j] = front_tbl[1][j]
    # compute j which has the maximum alignment score between seq1[len1/2 + 1, len1] and seq2[j, len2]
    for i in range(len1 // 2, len1):
        reverse_tbl[1][0] = (i - len1 // 2 + 1) * indel  # setting the first column
        for j in range(1, len2 + 1):
            # different from front_tbl, now we match characters from the end of seq1 and seq2
            pos1 = len1 - (i - len1 // 2) - 1
            pos2 = len2 - j
            cell1 = seq1[pos1]
            cell2 = seq2[pos2]
            similarity = cell_similarity(cell1, cell2)
            score1 = reverse_tbl[0][j - 1] + similarity
            score2 = reverse_tbl[0][j] + indel
            score3 = reverse_tbl[1][j - 1] + indel
            curr_score = max(score1, score2, score3)
            reverse_tbl[1][j] = curr_score
        # reset the table
        for j in range(0, len2 + 1):
            reverse_tbl[0][j] = reverse_tbl[1][j]
    max_j_pos = 0
    max_sum = front_tbl[1][0] + reverse_tbl[1][len2]
    for j in range(1, len2 + 1):
        curr_score = front_tbl[1][j] + reverse_tbl[1][len2 - j]
        if curr_score >= max_sum:
            max_sum = curr_score
            max_j_pos = j
    return max_j_pos


# pairwise alignment
#   seq1 and seq2 are lists of cells which could be passed to function cell_similarity
#   cell_similarity takes two cell classes as input and output a positive number;
#       this number is greater if the similarity is higher.
def alignment(seq1, seq2, cell_similarity=debug_similarity):
    indel = 0
    len1 = len(seq1)
    len2 = len(seq2)
    # base case
    if len1 == 0 and len2 == 0:
        return (0, [], [])
    elif len1 == 0:
        return (indel * len2, [-100] * len2, seq2)
    elif len2 == 0:
        return (indel * len1, seq1, [-100] * len1)

    elif len1 == 1:  # if seq1 contains only one cell
        max_sim = -1
        max_pos = -1
        for i in range(len2):
            curr_sim = cell_similarity(seq1[0], seq2[i])
            if curr_sim > max_sim:
                max_pos = i
                max_sim = curr_sim
        return (max_sim + (len2 - 1) * indel, [-100] * max_pos + seq1 + [-100] * (len2 - max_pos - 1), seq2)

    elif len2 == 1:  # if seq2 is a single character
        max_sim = -1
        max_pos = -1
        for i in range(len1):
            curr_sim = cell_similarity(seq1[i], seq2[0])
            if curr_sim > max_sim:
                max_pos = i
                max_sim = curr_sim
        return (max_sim + (len1 - 1) * indel, seq1, [-100] * max_pos + seq2 + [-100] * (len1 - max_pos - 1))

    # find the optimal j where we divide seq2
    j = find_optimal(seq1, seq2, cell_similarity)
    front = alignment(seq1[0: len1 // 2], seq2[0: j], cell_similarity)
    back = alignment(seq1[len1 // 2: len1], seq2[j: len2], cell_similarity)
    return (front[0] + back[0], front[1] + back[1], front[2] + back[2])


# Helper for multiple alignment
# align seq1 and seq2 in O(n) time with one requirement:
#   if remove all -100, seq1 and seq2 must be the same
#   align seq1 and seq2 by inserting * as indel, and output a tuple of the new alignment
def same_seq_alignment(seq1, seq2):
    len1 = len(seq1)
    len2 = len(seq2)
    i = 0
    j = 0
    new_seq1 = []
    new_seq2 = []
    while i != len1 or j != len2:
        if i == len1:
            new_seq1 += [-99] * (len2 - j)
            new_seq2 += seq2[j:]
            break
        elif j == len2:
            new_seq1 += seq1[i:]
            new_seq2 += [-99] * (len1 - i)
            break
        # both i and j not equals to len
        if seq1[i] == seq2[j]:

            new_seq1 += [seq1[i]]
            new_seq2 += [seq2[j]]
            i += 1
            j += 1
        elif seq1[i] == -100:
            new_seq1 += [-100]
            new_seq2 += [-99]
            i += 1
        elif seq2[j] == -100:
            new_seq1 += [-99]
            new_seq2 += [-100]
            j += 1
    return (new_seq1, new_seq2)


# Helper for multiple alignment
# seq1 and seq2 might has a mix of - and * as indel
#   align seq1 and seq2 by inserting # as indel, and output a tuple of the new alignment
def diff_seq_alignment(seq1, seq2):
    len1 = len(seq1)
    len2 = len(seq2)
    i = 0
    j = 0
    new_seq1 = []
    new_seq2 = []
    while i != len1 or j != len2:
        if i == len1:
            new_seq1 += [-98] * (len2 - j)
            new_seq2 += seq2[j:]
            break
        elif j == len2:
            new_seq1 += seq1[i:]
            new_seq2 += [-98] * (len1 - i)
            break
        # both i and j not equals to len
        if (seq1[i] != -99 and seq2[j] != -99) or (seq1[i] == -99 and seq2[j] == -99):
            new_seq1 += [seq1[i]]
            new_seq2 += [seq2[j]]
            i += 1
            j += 1
        elif seq1[i] == -99:
            new_seq1 += [-99]
            new_seq2 += [-98]
            i += 1
        elif seq2[j] == -99:
            new_seq1 += [-98]
            new_seq2 += [-99]
            j += 1
    return (new_seq1, new_seq2)


# print(diff_seq_alignment('a*b**', 'a-****'))

# multiple alignment
def mult_alignment(seq_list, cell_similarity=similarity_func2):

    nb_num = len(seq_list)
    # create a min heap for the similarity of all possible pairs - O(m^2 * n^2)
    heap_list = []
    for i in range(nb_num):
        for j in range(i + 1, nb_num):
            curr_align = alignment(seq_list[i], seq_list[j], cell_similarity)
            heap_list.append(
                (-curr_align[0], curr_align[1], curr_align[2], i, j))  # changing sign makes it into a max heap

    heapq.heapify(heap_list)

    # recurrently get the pair with the greatest similarity, merge pair, and remove from the heap
    alignment_group = []  # the groups which each alignment belongs in
    group_counter = 0  # increment each time there is a new group
    alignment_result = []  # to be returned
    for i in range(nb_num):
        alignment_result.append(None)
        alignment_group.append(-1)

    while (1):

        try:
            curr = heapq.heappop(heap_list)
            if DEBUG:
                print(curr)
        except:
            break
        i = curr[3]
        j = curr[4]



        if alignment_result[i] == None and alignment_result[j] == None:  # a new group
            if DEBUG:
                print("c1")
            alignment_group[i] = group_counter
            alignment_group[j] = group_counter
            group_counter += 1
            alignment_result[i] = curr[1]
            alignment_result[j] = curr[2]

        elif alignment_result[i] != None and alignment_result[j] != None:
            # check if this two sequences are in the same group or not;
            #   if they are, then the merging is done
            #   else, merge the two groups together
            if alignment_group[i] == alignment_group[j]:  # in the same group
                if DEBUG:
                    print("c2")
                continue
            else:
                if DEBUG:
                    print("c3")
                g1 = alignment_group[i]
                g2 = alignment_group[j]
                # align notebook indexed by i
                align1 = alignment_result[i]
                align2 = curr[1]
                new_alignment_i = same_seq_alignment(align1, align2)

                # align notebook indexed by j
                align1 = alignment_result[j]
                align2 = curr[2]
                new_alignment_j = same_seq_alignment(align1, align2)

                # align notebook i and j again
                final_alignment = diff_seq_alignment(new_alignment_i[0], new_alignment_j[0])

                # update all in g1 according to final_alignment
                for n in range(nb_num):
                    if alignment_group[n] == g1:
                        # need update
                        updated_member = []
                        mp = 0
                        for p in final_alignment[0]:  # alignment for i
                            if p == -98 or p == -99:
                                updated_member += [-100]
                            else:
                                updated_member += [alignment_result[n][mp]]
                                mp += 1
                        if mp != len(alignment_result[n]):  # shouldn't go here
                            print("buggy code: length doesn't match")
                            exit(1)
                        alignment_result[n] = updated_member

                # update all in g2 according to final_alignment
                for n in range(nb_num):
                    if alignment_group[n] == g2:
                        # need update
                        updated_member = []
                        mp = 0
                        for p in final_alignment[1]:  # alignment for j
                            if p == -98 or p == -99:
                                updated_member += [-100]
                            else:
                                updated_member += [alignment_result[n][mp]]
                                mp += 1
                        if mp != len(alignment_result[n]):  # shouldn't go here
                            print("buggy code: length doesn't match")
                            exit(1)
                        alignment_result[n] = updated_member
                        alignment_group[n] = g1

        else:  # only one of the sequences in the pair has been merged with other sequence(s)
            g1 = alignment_group[i]
            g2 = alignment_group[j]
            if g2 == -1:  # j is new_member
                if g1 == -1:  # shouldn't go here
                    print("buggy code!")
                    exit(1)
                if DEBUG:
                    print('c4')
                alignment_result[j] = curr[2]
                alignment_group[j] = g1
                new_member = j
                align1 = alignment_result[i]
                align2 = curr[1]
            else:  # i is new_member
                if g1 != -1:  # shouldn't go here
                    print("buggy code!")
                    exit(1)
                if DEBUG:
                    print('c5')
                alignment_result[i] = curr[1]
                alignment_group[i] = g2
                new_member = i
                align1 = alignment_result[j]
                align2 = curr[2]

            new_alignment = same_seq_alignment(align1, align2)

            # update new_member according to the new_alignment
            updated_member = []
            mp = 0
            for k in new_alignment[1]:  # align2
                if k == -99:
                    updated_member += [-100]
                else:
                    updated_member += [alignment_result[new_member][mp]]
                    mp += 1
            if mp != len(alignment_result[new_member]):
                print("buggy code: length doesn't match")
                exit(1)
            alignment_result[new_member] = updated_member

            # update all other alignments
            for n in range(nb_num):
                if n != new_member and alignment_group[n] == alignment_group[new_member]:
                    updated_member = []
                    mp = 0
                    for p in new_alignment[0]:  # align1
                        if p == -99:
                            updated_member += [-100]
                        else:
                            updated_member += [alignment_result[n][mp]]
                            mp += 1
                    if mp != len(alignment_result[n]):
                        print("buggy code: length doesn't match")
                        exit(1)
                    alignment_result[n] = updated_member

    return alignment_result

#
# if __name__ == '__main__':
#     start = timeit.default_timer()
#     seq_list = []
#
#     # read from iostream; comment out if want to read from file
#     print("Usage: \nnum of lines\ncell1\ncell2...")
#     num = int(sys.stdin.readline().strip('\n'))
#     for i in range(num):
#         s = sys.stdin.readline().strip('\n')
#         seq_list.append(s)


    #---------------------------------------------------------
    # read from file; comment out if want to read from input

    # if len(sys.argv) != 3:
    #     print("Usage: python3 mult_alignment.py input_file output_file")
    #     exit(1)
    # input_name = sys.argv[1]
    # output_name  = sys.argv[2]
    # f = open(input_name, 'r')
    # while (True):
    #     l = f.readline()
    #     if not l:
    #         break
    #     seq_list.append(l.rstrip())

    #----------------------------------------------------------

    #
    # print(seq_list)
    # alignment_result = mult_alignment(seq_list, similarity_func2)
    #
    # # print to output; comment out if want to write to file
    #
    # for i in alignment_result:
    #     print(i)


    #----------------------------------------------------------
    # write to file; comment out if want to print

    # fw = open(output_name, 'w')
    # for i in alignment_result:
    #     fw.write(i)
    #     fw.write('\n')
    # fw.close()

    #-----------------------------------------------------------

    # stop = timeit.default_timer()
    # print('Time: ', stop - start)


