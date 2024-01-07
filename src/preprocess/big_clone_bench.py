import os
import pickle
import random

import javalang
import psycopg2
from tqdm import tqdm


def generate_pairs(source_data_dir, target_data_dir):
    def read_code(type, name, startline, endline):
        with open(os.path.join(source_data_dir, type, name), 'r', encoding='utf-8') as fc:
            try:
                ret = ''.join(fc.readlines()[startline - 1:endline])
                javalang.parser.Parser(
                    javalang.tokenizer.tokenize(
                        ret
                    )
                ).parse_member_declaration()
                return ret
            except:
                return None

    def label(st, sl):
        return st if st <= 2 else (3 if sl >= 0.7 else (4 if 0.5 <= sl < 0.7 else 5))

    function_ids = set()
    with open('bcb_function_ids.txt', 'r', encoding='utf-8') as f:
        for line in f:
            function_ids.add(int(line.strip()))

    # connect database
    connection = psycopg2.connect(database='BigCloneBench', user='postgres', password='1234', port='5432')

    # access clones
    tnse2idx, idx2code = {}, {}
    clones = set()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT a.type, a.name, a.startline, a.endline, 
               b.type, b.name, b.startline, b.endline,
               c.syntactic_type, c.similarity_line, c.function_id_one, c.function_id_two
        FROM clones as c, functions as a, functions as b
        WHERE a.id = c.function_id_one and b.id = function_id_two
    """)
    for (type1, name1, startline1, endline1, type2, name2, startline2, endline2,
         syntactic_type, similarity_line, id1, id2) in tqdm(cursor.fetchall(), desc='Clones'):
        if id1 not in function_ids or id2 not in function_ids:
            continue
        a = (type1, name1, startline1, endline1)
        b = (type2, name2, startline2, endline2)
        if a not in tnse2idx:
            code1 = read_code(*a)
            if code1 is None:
                continue
            tnse2idx[a] = len(tnse2idx)
            idx2code[tnse2idx[a]] = code1
        if b not in tnse2idx:
            code2 = read_code(*b)
            if code2 is None:
                continue
            tnse2idx[b] = len(tnse2idx)
            idx2code[tnse2idx[b]] = code2
        clones.add((tnse2idx[a], tnse2idx[b], label(syntactic_type, similarity_line)))

    # access non-clones
    non_clones = set()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT a.type, a.name, a.startline, a.endline, 
               b.type, b.name, b.startline, b.endline,
               c.syntactic_type, c.similarity_line, c.function_id_one, c.function_id_two
        FROM false_positives as c, functions as a, functions as b
        WHERE a.id = c.function_id_one and b.id = function_id_two
    """)
    for (type1, name1, startline1, endline1, type2, name2, startline2, endline2,
         syntactic_type, similarity_line, id1, id2) in tqdm(cursor.fetchall(), desc='Non-clones'):
        if id1 not in function_ids or id2 not in function_ids:
            continue
        a = (type1, name1, startline1, endline1)
        b = (type2, name2, startline2, endline2)
        if a not in tnse2idx:
            code1 = read_code(*a)
            if code1 is None:
                continue
            tnse2idx[a] = len(tnse2idx)
            idx2code[tnse2idx[a]] = code1
        if b not in tnse2idx:
            code2 = read_code(*b)
            if code2 is None:
                continue
            tnse2idx[b] = len(tnse2idx)
            idx2code[tnse2idx[b]] = code2
        non_clones.add((tnse2idx[a], tnse2idx[b], -label(syntactic_type, similarity_line)))

    # disconnect database
    connection.close()

    # divide into train, val, and test set
    indices = list(range(len(idx2code)))
    random.seed(5)
    random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[1000:], indices[:500], indices[500:1000]

    # save
    pair_labels = {(pair_label[0], pair_label[1]): pair_label[2] for pair_label in list(clones) + list(non_clones)}
    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)
    with open(os.path.join(target_data_dir, 'codes.pkl'), 'wb') as f:
        for i in tqdm(range(len(idx2code)), desc='code'):
            pickle.dump(idx2code[i], f)
    for part, ids in zip(('train', 'val', 'test'), (train_indices, val_indices, test_indices)):
        with open(os.path.join(target_data_dir, f'{part}.pkl'), 'wb') as f:
            samples1, samples2 = [], []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if (ids[i], ids[j]) in pair_labels or (ids[j], ids[i]) in pair_labels:
                        label = pair_labels[(ids[i], ids[j])] if (ids[i], ids[j]) in pair_labels else pair_labels[
                            (ids[j], ids[i])]
                        samples1.append(((ids[i], ids[j]), label))
                    else:
                        samples2.append(((ids[i], ids[j]), -5))
            random.seed(1)
            if part == 'train':
                samples = list(random.sample(samples1, k=round(0.025 * len(samples1)))) + list(
                    random.sample(samples2, k=round(0.025 * len(samples2))))
            elif part == 'val':
                samples = samples1 + samples2
            else:
                samples = samples1
            for x in samples:
                pickle.dump(x, f)
            from collections import Counter
            print(Counter([x[1] for x in samples]), len(samples))
