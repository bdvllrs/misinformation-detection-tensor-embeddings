import pickle
from utils.dataloaders import DataLoader
from utils import get_fullpath


class StatementsLoader(DataLoader):
    def load(self):
        with open(get_fullpath(self.config.dataset.sentence_based.dataset_path), "rb") as file:
            statements = pickle.load(file)
        for k, statement in enumerate(statements):
            label = statement['label']
            if label not in self.articles.keys():
                self.articles[label] = []
            content = self._get_content(str(k) + "_statements", statement['text'], label=label, split_sentences=False)
            self.articles[label].append({
                "content": content
            })
        return self.articles, self.original_articles, self.vocabulary, self.frequency

