import json
from sklearn.model_selection import train_test_split


def json_to_iob_txt(json_data, out_filename):

    with open(out_filename, 'w') as f:
        for sent in json_data:
            for token in sent:
                word = token[0][0].encode('utf-8')
                tag = token[1].encode('utf-8')
                # print word, tag
                f.write(('%s %s\n') % (word, tag))
            f.write('\n')

    return


def main():
    with open('data/data.json', 'r') as f:
        dataset = json.load(f)

    print len(dataset)

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=666)
    test_a, test_b = train_test_split(test_data, test_size=0.5, random_state=666)

    json_to_iob_txt(train_data, 'data/eng.train.iob')
    json_to_iob_txt(test_a, 'data/eng.testa.iob')
    json_to_iob_txt(test_b, 'data/eng.testb.iob')
    return


if __name__ == '__main__':
    main()
