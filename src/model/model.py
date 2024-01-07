import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, token_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_gpu = torch.cuda.is_available()
        self.embedding_dim = embedding_dim

        self.embed = nn.Embedding(token_vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.combine = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 1)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.node_list = []
        self.batch_node = None

    def forward(self, x1, x2):
        def encode_subtree_roots(subtree_roots):
            def traverse(node, batch_index):
                size = len(node)
                if not size:
                    return None

                index, children_index = [], []
                current_node, children = [], []
                for i in range(size):
                    index.append(i)
                    current_node.append(node[i][0])
                    temp = node[i][1:]
                    c_num = len(temp)
                    for j in range(c_num):
                        children_index.append(i)
                        children.append(temp[j])

                th = torch.cuda if self.use_gpu else torch
                batch_current = self.linear(self.embed(th.LongTensor(current_node)))

                if children_index:
                    tree = traverse(children, [batch_index[i] for i in children_index])
                    batch_current.index_add_(0, th.LongTensor(children_index), tree)

                self.batch_node = self.batch_node.index_reduce(0, th.LongTensor(batch_index), batch_current, 'amax')
                return batch_current

            batch_size = len(subtree_roots)
            self.batch_node = torch.zeros(batch_size, self.embedding_dim, requires_grad=True).cuda()
            traverse(subtree_roots, list(range(batch_size)))
            return self.batch_node

        def encode(x):
            batch_subtrees = x
            batch_size, n_subtrees_list = len(batch_subtrees), [len(subtrees) for subtrees in batch_subtrees]

            all_subtrees = [batch_subtrees[i][j] for i in range(batch_size) for j in range(n_subtrees_list[i])]
            batch_subtree_embeddings = encode_subtree_roots(all_subtrees)
            batch_subtree_embeddings = torch.split(batch_subtree_embeddings, n_subtrees_list)

            batch_subtree_embeddings = nn.utils.rnn.pad_sequence(batch_subtree_embeddings)
            batch_subtree_embeddings = nn.utils.rnn.pack_padded_sequence(
                batch_subtree_embeddings, torch.tensor(n_subtrees_list), enforce_sorted=False)
            output, _ = self.rnn(batch_subtree_embeddings)
            batch_subtree_embeddings, _ = nn.utils.rnn.pad_packed_sequence(output, padding_value=0)
            batch_subtree_embeddings = self.combine(batch_subtree_embeddings)
            batch_subtree_embeddings = torch.transpose(batch_subtree_embeddings, 0, 1)
            batch_subtree_embeddings = [subtree_embeddings[:n_subtrees] for n_subtrees, subtree_embeddings in
                                        zip(n_subtrees_list, batch_subtree_embeddings)]
            return batch_subtree_embeddings

        h1, h2 = encode(x1), encode(x2)

        z11, z12 = [torch.max(hi1, dim=0).values for hi1 in h1], [torch.max(hi2, dim=0).values for hi2 in h2]
        z11, z12 = torch.stack(z11), torch.stack(z12)
        z1 = torch.sum(z11 * z12, dim=1)

        z2 = torch.stack(
            [(hi1.unsqueeze(0) * hi2.unsqueeze(1)).sum(dim=0).sum(dim=0) for hi1, hi2 in zip(h1, h2)]
        )
        z2 = self.linear2(z2).squeeze()

        return z1 + z2
