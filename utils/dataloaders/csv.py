from utils.dataloaders import DataLoader
import csv


class CSVLoader(DataLoader):
    def load(self):
        with open(self.config.Dataset.dataset_path, "rt", encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            data = next(reader)
            print("Import csv...")
            k = 0
            while data:
                try:
                    data = next(reader)
                except csv.Error:
                    break
                uid, title, author, text, label = data
                article_type = "fake" if int(label) else "real"
                self.articles[article_type].append({
                    'content': self._get_content(uid, text, type=article_type),
                    'title': self._get_content(uid + "_title", title, type=article_type)
                })
                k += 1
            print("Imported.")
        return self.articles, self.original_articles, self.vocabulary, self.frequency

