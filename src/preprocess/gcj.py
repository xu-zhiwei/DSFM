import os
import pickle
import random

import javalang


def generate_pairs(source_data_dir, target_data_dir):
    def read_code(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = []
                for line in f.readlines():
                    if any([line.strip()[:7] == 'package', line.strip()[:6] == 'import']):
                        continue
                    text.append(line)
                text = ''.join(text)
            tokens = javalang.tokenizer.tokenize(text)
            parser = javalang.parser.Parser(tokens)
            parser.parse_member_declaration()
            return text
        except:
            return None

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    total_code_categories, unique_codes = [], set()
    for root, dirs, files in os.walk(source_data_dir):
        for file in files:
            try:
                code = read_code(os.path.join(root, file))
                if code is not None:
                    if code not in unique_codes:
                        total_code_categories.append((code, os.path.basename(root)))
                    unique_codes.add(code)
            except UnicodeDecodeError:
                pass
    random.seed(1)
    random.shuffle(total_code_categories)

    code2idx = {}
    with open(os.path.join(target_data_dir, 'codes.pkl'), 'wb') as fc:
        for role, code_categories in zip(
                ['train', 'val', 'test'],
                [total_code_categories[:1000], total_code_categories[1000:1331], total_code_categories[1331:]]):
            with open(os.path.join(target_data_dir, f'{role}.pkl'), 'wb') as fp:
                for i in range(len(code_categories)):
                    for j in range(i + 1, len(code_categories)):
                        if code_categories[i][0] not in code2idx:
                            code2idx[code_categories[i][0]] = len(code2idx)
                            pickle.dump(code_categories[i][0], fc)
                        if code_categories[j][0] not in code2idx:
                            code2idx[code_categories[j][0]] = len(code2idx)
                            pickle.dump(code_categories[j][0], fc)
                        pickle.dump(((code2idx[code_categories[i][0]], code2idx[code_categories[j][0]]),
                                     1 if code_categories[i][1] == code_categories[j][1] else 0), fp)
