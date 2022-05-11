import csv
from operator import itemgetter

from fastpunct import FastPunct

import GRUEN.Main as gruen
from keybert import KeyBERT


def join_csv():
    with open('all_haiku.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        read_csv = list(reader)

    fieldnames = [*fieldnames, 'clean0', 'clean1', 'clean2']
    for row in read_csv:
        haiku_rows = [to_alpha_space_lower_strip(s) for s in itemgetter('0', '1', '2')(row)]
        row['clean0'] = haiku_rows[0]
        row['clean1'] = haiku_rows[1]
        row['clean2'] = haiku_rows[2]

    with open('lines.txt', 'r') as file:
        lines = file.readlines()

    haikus = []
    for index, line in enumerate(lines[:3]):
        haiku = line.replace('$', '').replace('\n', '').strip().split(' / ')
        clean_haiku = [to_alpha_space_lower_strip(line) for line in haiku]
        haikus.append(dict(zip(
            fieldnames,
            [index, *haiku, 'r/haiku', ''.join(e for e in ''.join(haiku) if e.isalpha()), *clean_haiku]
        )))

    with open('joined_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(read_csv)
        writer.writerows(haikus)


def to_alpha_space_lower_strip(s: str) -> str:
    return ''.join(filter(lambda s: s.isalpha() or s.isspace(), s)).lower().strip()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    join_csv()
    
    with open('joined_haikus.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        read_csv = list(reader)

    doc = []
    for row in read_csv:
        doc.extend(itemgetter('clean0', 'clean1', 'clean2')(row))
    fastpunct = FastPunct()
    punct = fastpunct.punct(doc)
    for row, chunk in zip(read_csv, chunks(punct, 3)):
        for k, v in zip(['punct0', 'punct1', 'punct2'], chunk):
            row[k] = v

    docs = [' '.join(row) for row in chunks(punct, 3)]
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs=docs, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=1)
    for row, keyword in zip(read_csv, [row[0][0] for row in keywords]):
        row['keywords'] = keyword

    for gruen_score, row in zip(gruen.get_gruen(docs), read_csv):
        row['gruen'] = gruen_score

    fieldnames = [*fieldnames, 'punct0', 'punct1', 'punct2', 'keywords', 'gruen']

    with open('processed_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(read_csv)
