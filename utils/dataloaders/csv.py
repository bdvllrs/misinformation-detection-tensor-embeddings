from utils.dataloaders import DataLoader
import csv


class CSVLoader(DataLoader):
    def load(self):
        with open(self.config.dataset.dataset_path, "rt", encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            data = next(reader)
            print("Import csv...")
            k = 0
            done = False
            while data and not done:
                if not k % 100:
                    print(k)
                try:
                    data = next(reader)
                except csv.Error:
                    done = True
                index_uid = self.config.dataset.csv.uid
                index_title = self.config.dataset.csv.title
                index_content = self.config.dataset.csv.content
                index_label = self.config.dataset.csv.label
                uid, title, text, label = data[index_uid], data[index_title], data[index_content], data[index_label]
                content = self._get_content(uid, text, label=label)
                if 10 < len(content) < 2000:
                    if label not in self.articles.keys():
                        self.articles[label] = []
                    title = self._get_content(uid + "_title", title, label=label)
                    self.articles[label].append({
                        'content': content,
                        'title': title
                    })
                    k += 1
            print("Imported.")
        return self.articles, self.original_articles, self.vocabulary, self.frequency
