from concurrent.futures import ThreadPoolExecutor

import numpy as np
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


def save_to_es(dataset='Assist2009'):
    with Elasticsearch(hosts=['http://localhost:9200/']).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
    ) as es:
        data_dir = f'./data/{dataset}/{dataset}.npz'
        with np.load(data_dir, allow_pickle=True) as data:
            y, skill, real_len = data['y'], data['skill'], data['real_len']

        users = list(range(len(real_len)))
        train_num = int(0.8 * len(users))
        train_test_split = {'train': users[:train_num], 'test': users[train_num:]}

        def f(index, users_mode):
            for sid in tqdm(users_mode):
                for rid in range(real_len[sid]):
                    yield {'_index': index, 'user': sid, 'loc': rid, 'skill': skill[sid][rid],
                           'y': y[sid][:rid + 1], 'history': ' '.join(skill[sid][:rid + 1].astype(str))}

        def index_mode(mode):
            index = f'{dataset}_{mode}'
            es.indices.delete(index=index)
            es.indices.create(index=index)
            users_mode = train_test_split[mode]
            actions = f(index, users_mode)
            helpers.bulk(client=es, actions=actions)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(index_mode, ('train', 'test'))
        print("Save to ES Finished!")


if __name__ == '__main__':
    # save_to_es('assist09')
    with Elasticsearch(hosts=['http://localhost:9200/']).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
    ) as es:
        index_ = 'assist09_train'
        result = es.search(index=index_)
        print(result)
        queries = []
        np.random.seed(1)
        skill = np.random.randint(1, 10, size=(20,))
        print(skill)
        query = [{},
                 {'size': 5,
                  'collapse': {'field': 'user'},
                  'query': {'bool': {
                      'filter':
                          [
                              {'term': {'skill': 6}},
                              {'bool': {'must_not': {'term': {'user': 23}}}}],
                      'must': {'match': {'history': ' '.join(skill.astype('str'))}}}},
                  '_source': ['user']}]
        queries.extend(query)
        skill = np.random.randint(1, 10, size=(20,))
        print(skill)
        query = [{},
                 {'size': 5,
                  'collapse': {'field': 'user'},
                  'query': {'bool': {
                      'filter':
                          [
                              {'term': {'skill': 6}},
                              {'bool': {'must_not': {'term': {'user': 2756}}}}],
                      'must': {'match': {'history': ' '.join(skill.astype('str'))}}}},
                  '_source': ['user']}]
        queries.extend(query)
        result = es.msearch(index=index_, searches=queries)['responses']
        hits = [_['hits']['hits'] for _ in result]
        for hit in hits:
            for rs in hit:
                print(rs['_score'], rs['_source']['user'])
            print()

