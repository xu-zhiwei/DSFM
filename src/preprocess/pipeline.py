import pickle
from pathlib import Path
from queue import Queue

import javalang
import networkx as nx
import pycparser
from tqdm import tqdm

from . import big_clone_bench
from . import gcj
from . import oj_clone


class Pipeline:
    def __init__(self, source_data_dir, target_data_dir, dataset_name):
        """
        :param source_data_dir: original preprocess directory.
        :param target_data_dir: target preprocess directory.
        :param dataset_name: name of dataset, i.e., OJClone, BigCloneBench, GCJ.
        """
        self.source_data_dir = Path(source_data_dir)
        self.target_data_dir = Path(target_data_dir)
        self.dataset_name = dataset_name

        self.delimiter = {
            'OJClone': ['<FuncDef>', '<FuncCall>', '<For>', '<While>', '<DoWhile>', '<Switch>', '<Case>', '<Default>',
                        '<If>', '<Return>'],
            'GCJ': ['<ClassDeclaration>', '<MethodDeclaration>', '<ForStatement>', '<WhileStatement>', '<DoStatement>',
                    '<IfStatement>', '<SwitchStatement>', '<SwitchStatementCase>', '<TryStatement>',
                    '<CatchClause>', ' <ReturnStatement>'],
        }
        self.delimiter['BigCloneBench'] = self.delimiter['GCJ']

    def run(self):
        """
        Runner for pipeline.
        """
        print('1. Generating pairs...')
        self.generate_pairs(self.source_data_dir, self.target_data_dir)
        print('2. Loading codes...')
        codes = self.load_codes()
        print('3. Parsing codes into ASTs...')
        asts = self.parse_codes(codes)
        print('4. Unifying ASTs...')
        asts = self.unify_asts(asts)
        print('5. Extracting subtrees...')
        subtrees = self.extract_subtrees(asts)
        print('6. Saving subtrees...')
        self.save_subtrees(subtrees)

    def generate_pairs(self, source_data_dir, target_data_dir):
        """
        Generate code pairs according to the input dataset.
        :param source_data_dir:
        :param target_data_dir:
        :return:
        """
        if self.dataset_name == 'OJClone':
            oj_clone.generate_pairs(source_data_dir, target_data_dir)
        elif self.dataset_name == 'GCJ':
            gcj.generate_pairs(source_data_dir, target_data_dir)
        elif self.dataset_name == 'BigCloneBench':
            big_clone_bench.generate_pairs(source_data_dir, target_data_dir)
        else:
            raise ValueError(f'Do not support `{self.dataset_name}`')

    def load_codes(self):
        """
        Load source code files.
        :return: List of index of code, List of codes.
        """
        code_file = self.target_data_dir / 'codes.pkl'
        codes = []
        with open(code_file, 'rb') as f:
            while True:
                try:
                    codes.append(pickle.load(f))
                except EOFError:
                    break
        return codes

    def parse_codes(self, codes):
        """
        Parse codes into ASTs.
        :param codes: List of codes.
        :return: List of ASTs.
        """
        asts = []
        for code in tqdm(codes):
            if self.dataset_name == 'OJClone':
                parser = pycparser.c_parser.CParser()
                ast = parser.parse(code)
            elif self.dataset_name in ('GCJ', 'BigCloneBench', 'BigCloneBenchT1', 'BigCloneBenchT2', 'BigCloneBenchST3',
                                       'BigCloneBenchMT3', 'BigCloneBenchWT3T4'):
                tokens = javalang.tokenizer.tokenize(code)
                parser = javalang.parser.Parser(tokens)
                ast = parser.parse_member_declaration()
            else:
                raise ValueError(f'{self.dataset_name} is not supported.')
            asts.append(ast)
        return asts

    def unify_asts(self, asts):
        """
        Unify ASTs.
        :param asts: List of ASTs to be unified.
        :return: List of ASTs.
        """

        def unify_ast_for_c(obj, g, par):
            if isinstance(obj, pycparser.c_ast.Node):
                g.add_node(g.number_of_nodes(), name=f'<{obj.__class__.__name__}>')
                if par is not None:
                    g.add_edge(par, g.number_of_nodes() - 1)
                par = g.number_of_nodes() - 1
                for key in obj.__slots__[:-2]:
                    value = getattr(obj, key)
                    if not value:
                        continue
                    unify_ast_for_c(value, g, par)
            elif isinstance(obj, list):
                for e in obj:
                    unify_ast_for_c(e, g, par)
            elif isinstance(obj, str):
                g.add_node(g.number_of_nodes(), name=obj)
                g.add_edge(par, g.number_of_nodes() - 1)
            else:
                raise RuntimeError(f'Unhandled type: {type(obj)}')

        def unify_ast_for_java(obj, g, par):
            if isinstance(obj, javalang.ast.Node):
                g.add_node(g.number_of_nodes(), name=f'<{obj.__class__.__name__}>')
                if par is not None:
                    g.add_edge(par, g.number_of_nodes() - 1)
                par = g.number_of_nodes() - 1
                for key in obj.attrs:
                    value = getattr(obj, key)
                    if not value or any([value == [None] * i for i in range(1, 10)]):
                        continue
                    unify_ast_for_java(value, g, par)
            elif isinstance(obj, list) or isinstance(obj, set):
                for e in obj:
                    unify_ast_for_java(e, g, par)
            elif isinstance(obj, str) or isinstance(obj, bool) or obj is None:
                g.add_node(g.number_of_nodes(), name=obj)
                g.add_edge(par, g.number_of_nodes() - 1)
            else:
                raise RuntimeError(f'Unhandled type: {type(obj)}, {obj}')

        unified_asts = []
        for ast in tqdm(asts):
            unified_ast = nx.DiGraph()
            if self.dataset_name == 'OJClone':
                unify_ast_for_c(ast, unified_ast, None)
            elif self.dataset_name in ('GCJ', 'BigCloneBench', 'BigCloneBenchT1', 'BigCloneBenchT2', 'BigCloneBenchST3',
                                       'BigCloneBenchMT3', 'BigCloneBenchWT3T4'):
                unify_ast_for_java(ast, unified_ast, None)
            else:
                raise ValueError(f'{self.dataset_name} is not supported.')
            unified_asts.append(unified_ast)
        return unified_asts

    def extract_subtrees(self, asts):
        def name(x, tree):
            return tree.nodes[x]['name']

        def dfs(root, tree, q, flag):
            ret = [name(root, tree)]
            if name(root, tree) in self.delimiter[self.dataset_name] and flag:
                q.put(root)
            else:
                for child in tree[root]:
                    ret.append(dfs(child, tree, q, True))
            return ret

        def extract_subtrees_from_ast(root, tree, q):
            ret = []
            if name(root, tree) in self.delimiter[self.dataset_name]:
                y = dfs(root, tree, q, False)
                ret.append(y)
            if q.empty():
                for child in tree[root]:
                    ret.extend(extract_subtrees_from_ast(child, tree, q))
            else:
                ret.extend(extract_subtrees_from_ast(q.get(), tree, q))
            return ret

        subtrees = []
        for ast in tqdm(asts):
            subtree = extract_subtrees_from_ast(0, ast, Queue())
            if not subtree:
                subtree = [dfs(0, ast, Queue(), False)]
            subtrees.append(subtree)
        return subtrees

    def save_subtrees(self, subtrees):
        subtree_file = self.target_data_dir / 'subtrees.pkl'
        with open(subtree_file, 'wb') as f:
            for subtree in subtrees:
                pickle.dump(subtree, f)
