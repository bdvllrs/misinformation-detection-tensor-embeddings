import numpy as np
from utils.postprocessing import PostProcessing
from utils import embedding_matrix_2_kNN
from utils.Config import Config
from utils.ArticlesProvider import ArticlesProvider


class SelectLabelsPostprocessor(PostProcessing):
    """
    Select smartly the articles to label to optimize the result
    """

    def __init__(self, config: Config, articles: ArticlesProvider):
        super().__init__(config, articles)

    def apply(self, tensor):
        graph = embedding_matrix_2_kNN(tensor, k=self.config.graph.num_nearest_neighbours, mode="distance").toarray()
        components = self.connected_components(graph)
        labels = [0 for _ in self.articles.labels]
        for k, neighbors in components.items():
            total_weight = 0
            weights_nodes = {}
            for node in neighbors:
                for node2, d in enumerate(graph[node]):
                    if d > 0:  # it's a neighbor
                        total_weight += d
                        if node2 in weights_nodes.keys():
                            weights_nodes[node2] += d
                        else:
                            weights_nodes[node2] = d
            k_max_weight = np.argmax(1/total_weight * np.array(list(weights_nodes.values()))) if total_weight > 0 else neighbors[0]
            labels[k_max_weight] = self.articles.labels_untouched[k_max_weight]
        self.articles.labels = labels

    @staticmethod
    def dfs(graph, origin, visited=None):
        if visited is None:
            visited = [origin]
        for i, d in enumerate(graph[origin]):
            if d != 0 and i not in visited:
                visited.append(i)
                visited = SelectLabelsPostprocessor.dfs(graph, i, visited)
        return visited

    @staticmethod
    def connected_components(graph):
        components = {}
        k = 0
        for i, d in enumerate(graph[0]):
            if i not in components.keys():
                for j in SelectLabelsPostprocessor.dfs(graph, i):
                    if j in components.keys():
                        components[j].append(k)
                    else:
                        components[j] = [k]
                k += 1
        return components

