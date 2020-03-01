import gensim
import networkx as nx
import numpy as np

from loader import load

# datasets = ['cora', 'citeseer', 'pubmed']
datasets = ['cora']
# datasets = ['slashdot']
recall_outs = []
for dataset in datasets:
    for K in [5, 10, 15, 20]:
        train_data, test_data = load(dataset)[:2]
        train_links = train_data.transpose(0, 1).numpy()
        test_links = test_data.transpose(0, 1).numpy()
        trainG = nx.DiGraph()
        testG = nx.DiGraph()
        trainG.add_edges_from(list(train_links))
        testG.add_edges_from(list(test_links))
        recall_in = []
        recall_out = []
        # print('loading embedding file')
        embedding = gensim.models.KeyedVectors.load_word2vec_format(
            './output/{}/I/embeddings.{}.pt.txt'.format(dataset, dataset))
        # index = gensim.similarities.MatrixSimilarity(gensim.matutils.Dense2Corpus(embedding.vectors.T))
        # print(list(index))
        # exit()
        # print('evluating')
        for (src, dst) in testG.edges:
            if str(src) not in embedding.vocab:
                continue
            m = 0
            n = 0
            for (dst_, p) in embedding.similar_by_word(str(src), topn=100):
                if not trainG.has_edge(src, int(dst_)):
                    n = n + 1
                    if n >= K:
                        break
                    if testG.has_edge(src, int(dst_)):
                        m = m + 1
            recall_out.append(m * 1.0 / K)
        print('{} recall_out@{}: {:.4f}'.format(dataset, K, np.mean(recall_out)))

        # for (src, dst) in testG.edges:
        #     if str(dst) not in embedding.vocab:
        #         continue
        #     m = 0
        #     n = 0
        #     for (src_, p) in embedding.wv.similar_by_word(str(src), topn=100):
        #         if not trainG.has_edge(int(src_), dst):
        #             n = n + 1
        #             if n > K:
        #                 break
        #             if testG.has_edge(int(src_), dst):
        #                 m = m + 1
        #     recall_in.append(m*1.0/n)
# for x in recall_outs:
#     print('{:.4f}'.format(x))
