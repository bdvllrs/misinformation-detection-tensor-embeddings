from utils import Config
from utils.ArticlesProvider import ArticlesProvider
from utils.decomposition import Decomposition
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class LDADecomposition(Decomposition):
    def __init__(self, config: Config, articles: ArticlesProvider):
        super().__init__(config, articles)
        self.lda = LatentDirichletAllocation(n_components=config.embedding.rank_parafac_decomposition, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=1000,
                                        stop_words='english')

    def apply(self):
        articles = [article['content'] for article in self.articles.articles['fake']] + [article['content'] for article
                                                                                         in
                                                                                     self.articles.articles['real']]
        all_articles_concatenate = [' '.join(text) for text in articles]
        tf = self.tf_vectorizer.fit_transform(all_articles_concatenate)
        self.lda.fit(tf)
        return (self.lda.transform(tf))


