import os
import requests
from bs4 import BeautifulSoup
import pickle

types = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
url_base = "https://www.politifact.com/truth-o-meter/rulings/"
file_path = os.path.abspath('../../Dataset/Politifact/dataset.pkl')

dataset = []


def get_page(ruling_type, page_num):
    url = url_base + ruling_type + "/?page=" + str(page_num)
    content = requests.get(url).text
    soup = BeautifulSoup(content, 'html.parser')
    statements = []
    for statement in soup.find_all('div', attrs={'class': 'statement'}):
        date = statement.find('span', attrs={"class": "article__meta"}).text
        statements.append({
            'text': statement.find('p', attrs={"class": "statement__text"}).text.replace('\n', '').replace(u'\xa0',
                                                                                                           u' '),
            'source': statement.find('p', attrs={"class": "statement__source"}).text,
            'date': date,
            'label': ruling_type
        })
    return statements


if __name__ == "__main__":
    for p in range(1, 51):
        print("Page", p)
        for ruling_type in types:
            dataset.extend(get_page(ruling_type, p))

    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)
