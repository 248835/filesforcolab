import csv
from operator import itemgetter

from fastpunct import FastPunct
from tqdm import tqdm

import GRUEN.Main as GRUEN
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
    for row in read_csv:
        haiku_rows = [to_alpha_space_lower_strip(s) for s in itemgetter('0', '1', '2')(row)]
        if len(haiku_rows[0]) > 100 or len(haiku_rows[1]) > 100 or len(haiku_rows[2]) > 100:
            continue
        haikus.append(row)

    for index, line in enumerate(lines):
        haiku = line.replace('$', '').replace('\n', '').strip().split(' / ')
        clean_haiku = [to_alpha_space_lower_strip(line) for line in haiku]
        haikus.append(dict(zip(
            fieldnames,
            [index, *haiku, 'r/haiku', ''.join(e for e in ''.join(haiku) if e.isalpha()).upper(), *clean_haiku]
        )))

    with open('joined_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(haikus)


def to_alpha_space_lower_strip(s: str) -> str:
    return ''.join(filter(lambda s: s.isalpha() or s.isspace(), s)).lower().strip()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_punct():
    with open('joined_haikus.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        read_csv = list(reader)

    fieldnames = [*fieldnames, 'punct0', 'punct1', 'punct2']

    with open('punct_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    fastpunct = FastPunct()
    for chunk in tqdm(list(chunks(read_csv, 500))):
        cleans = [itemgetter('clean0', 'clean1', 'clean2')(row) for row in chunk]
        flat_cleans = [item for sublist in cleans for item in sublist]

        punct = fastpunct.punct(flat_cleans)

        for row, puncts in zip(chunk, chunks(punct, 3)):
            row['punct0'] = puncts[0]
            row['punct1'] = puncts[1]
            row['punct2'] = puncts[2]

        with open('punct_haikus.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(chunk)


def get_keywords():
    with open('punct_haikus.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        read_csv = list(reader)

    punct = [itemgetter('clean0', 'clean1', 'clean2')(row) for row in read_csv]
    docs = [' '.join(row) for row in punct]

    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs=docs, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=1)
    for row, keyword in zip(read_csv, [row[0][0] for row in keywords]):
        row['keywords'] = keyword
    fieldnames = [*fieldnames, 'keywords']

    with open('keyword_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(read_csv)


def get_gruen():
    with open('keyword_haikus.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        read_csv = list(reader)
    punct = [itemgetter('punct0', 'punct1', 'punct2')(row) for row in read_csv]
    docs = [' '.join(row) for row in punct]
    for gruen_score, row in zip(GRUEN.get_gruen(docs), read_csv):
        row['GRUEN'] = gruen_score
    fieldnames = [*fieldnames, 'GRUEN']
    with open('processed_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(read_csv)


if __name__ == '__main__':
    # join_csv()

    # get_punct()

    # get_keywords()

    get_gruen()
