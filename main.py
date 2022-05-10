import csv
from operator import itemgetter

from fastpunct import FastPunct
from tqdm import tqdm

import GRUEN.Main as gruen
from keybert import KeyBERT
from phonemizer import phonemize
from phonemizer.separator import Separator


def join_csv():
    with open('all_haiku.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        read_csv = list(reader)

    with open('lines.txt', 'r') as file:
        lines = file.readlines()

    haikus = []
    for index, line in enumerate(lines):
        haiku = line.replace('$', '').replace('\n', '').strip().split(' / ')
        haikus.append(dict(zip(
            fieldnames,
            [index, *haiku, 'r/haiku', ''.join(e for e in ''.join(haiku) if e.isalnum())]
        )))

    with open('joined_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(read_csv)
        writer.writerows(haikus)


def to_alpha_space_lower_strip(s: str) -> str:
    return ''.join(filter(lambda s: s.isalpha() or s.isspace(), s)).lower().strip()


if __name__ == '__main__':
    with open('joined_haikus.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        read_csv = list(reader)

    fastpunct = FastPunct()
    kw_model = KeyBERT()
    for row in tqdm(read_csv[:50]):
        haiku_rows = [to_alpha_space_lower_strip(s) for s in itemgetter('0', '1', '2')(row)]
        punct = fastpunct.punct(haiku_rows)
        for k, v in zip(['punct0', 'punct1', 'punct2'], punct):
            row[k] = v
        keywords = kw_model.extract_keywords(' '.join(punct), keyphrase_ngram_range=(1, 2), stop_words=None)
        row['keywords'] = keywords[0][0]

    doc = [' '.join(itemgetter('punct0', 'punct1', 'punct2')(row)) for row in read_csv[:50]]
    for gruen_score, row in zip(gruen.get_gruen(doc), read_csv[:50]):
        row['gruen'] = gruen_score

    print(read_csv[:50])
    fieldnames = [*fieldnames, 'punct0', 'punct1', 'punct2', 'keywords', 'gruen']

    with open('processed_haikus.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(read_csv)
