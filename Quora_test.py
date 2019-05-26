import numpy as nmp
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
import pandas as pnd


class LoadData:
    @staticmethod
    def load_csv(path: str):
        datas = pnd.read_csv(path)
        return datas

    @staticmethod
    def prepare_corpus(loaded_csv: pnd.DataFrame):
        def correct_text(get_text: str):
            result_text = get_text.replace('?', ' vopros ')
            result_text = result_text.replace(',', ' zapat ')
            result_text = result_text.replace('.', ' toch ')
            result_text = result_text.replace(':', ' dwoet ')
            result_text = result_text.replace('(', ' lev_skob ')
            result_text = result_text.replace(')', ' pra_skob ')
            result_text = result_text.replace('  ', ' ')
            result_text = result_text.lower()
            return result_text

        loaded_csv = loaded_csv.fillna('empty')

        first_row = list(loaded_csv['question1'].dropna().values)
        second_row = list(loaded_csv['question2'].dropna().values)

        first_row = list(map(correct_text, first_row))
        second_row = list(map(correct_text, second_row))

        return first_row, second_row

    @staticmethod
    def tokenize_corpus(corpus):
        tokens_list = [x.split() for x in corpus]
        return tokens_list

    @staticmethod
    def vector_corpus(tokenized_corpus, vocab):
        new_corpus = []

        for sentence in tokenized_corpus:
            while 'nan' in sentence:
                sentence.remove('nan')
            while 'null' in sentence:
                sentence.remove('null')
            if len(sentence) == 0:
                sentence.append('empty')
            try:
                new_corpus.append([vocab[word] for word in sentence])
            except KeyError:
                loc_app = []
                for word in sentence:
                    if word in vocab:
                        loc_app.append(vocab[word])
                    else:
                        continue
                if len(loc_app) > 0:
                    new_corpus.append(loc_app)
                else:
                    loc_app.append(vocab['empty'])
                    new_corpus.append(loc_app)

        return new_corpus

    @staticmethod
    def load_dense_vocab():
        raw_weis = nmp.load('Result_embeddings2.npy')
        if raw_weis.shape[1] > 1000:
            raw_weis = nmp.rot90(raw_weis)
        return raw_weis

    @staticmethod
    def vector_sent(arr_of_wordvect: list, dense_vocab: dict):
        new_array = []
        for line in arr_of_wordvect:
            word_vectors = [dense_vocab[word] for word in line]
            line_vector = sum(word_vectors)
            new_array.append(line_vector)

        return new_array

    @staticmethod
    def get_test_data():
        print('Загрузка CSV')
        loaded_data = LoadData.load_csv('test.csv')
        print('Подготовка корпуса')
        first_q, second_q = LoadData.prepare_corpus(loaded_data)
        del loaded_data

        raw_dict = nmp.load('idx2w.npy')
        word2idx = {w: idx for (idx, w) in enumerate(raw_dict)}
        del raw_dict

        dense_vocab = LoadData.load_dense_vocab()

        slice_len = 100000
        for num, slice_range in enumerate(range(0, len(first_q), slice_len)):
            first_q_part = first_q[slice_range:slice_range + slice_len]
            second_q_part = second_q[slice_range:slice_range + slice_len]

            print('Токенизация', num)
            first_q_token = LoadData.tokenize_corpus(first_q_part)
            second_q_token = LoadData.tokenize_corpus(second_q_part)
            del first_q_part
            del second_q_part

            print('Векторизация', num)
            first_q_vect = LoadData.vector_corpus(first_q_token, word2idx)
            second_q_vect = LoadData.vector_corpus(second_q_token, word2idx)
            del first_q_token
            del second_q_token

            first_q_res = LoadData.vector_sent(first_q_vect, dense_vocab)
            second_q_res = LoadData.vector_sent(second_q_vect, dense_vocab)

            print('Сохранение данных', num)
            nmp.save('quora_test/q1_' + str(num), first_q_res)
            nmp.save('quora_test/q2_' + str(num), second_q_res)

    @staticmethod
    def load_ready_test_data(part):
        first_q_res = nmp.load('quora_test/q1_{}.npy'.format(part))
        second_q_res = nmp.load('quora_test/q2_{}.npy'.format(part))

        return first_q_res, second_q_res


class AnsPredict:
    def __init__(self, embedding_len):
        self.input_ph_q1 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])
        self.input_ph_q2 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])

        dataset = tf.data.Dataset.from_tensor_slices((self.input_ph_q1, self.input_ph_q2)).cache()
        dataset = dataset.batch(50)
        self.iterate = dataset.make_initializable_iterator()
        input_batch_q1, input_batch_q2 = self.iterate.get_next()

        dim_dist = input_batch_q1 - input_batch_q2

        lay_q1 = tf.layers.dense(inputs=input_batch_q1,
                                 units=embedding_len,
                                 activation=tf.nn.sigmoid)

        lay_q2 = tf.layers.dense(inputs=input_batch_q2,
                                 units=embedding_len,
                                 activation=tf.nn.sigmoid)

        lay_q1 = lay_q1 * input_batch_q1
        lay_q2 = lay_q2 * input_batch_q2

        lay_q = lay_q1 * lay_q2 * dim_dist

        lay_1_flat = lay_q

        lay_2 = tf.layers.dense(inputs=lay_1_flat,
                                units=3000,
                                activation=tf.nn.relu)

        lay_3 = tf.layers.dense(inputs=lay_2,
                                units=1,
                                activation=tf.nn.sigmoid)

        self.lay_3 = tf.reduce_mean(lay_3, 1)

    def test_net(self, loc_dataset):
        sv = tf.Session()
        with sv as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver.restore(sv, 'C:/TF/vk/save' + '/ckp')

            sess.run(self.iterate.initializer, feed_dict={self.input_ph_q1: loc_dataset[0],
                                                          self.input_ph_q2: loc_dataset[1]})
            list_ans = []

            limit = int(nmp.ceil(len(loc_dataset[0]) / 50))
            gs = 0
            while gs < limit - 1:
                for _ in tqdm(range(limit),
                              total=limit,
                              ncols=100,
                              leave=False,
                              unit='b'):
                    s = sess.run(self.lay_3)
                    list_ans.append(s)
                    gs += 1

        print("Готово")
        return list_ans

    @staticmethod
    def get_log_directory():
        """Создаёт путь для логов с учётом времени начала работы"""
        right_now_time = datetime.now()
        logdir = '{0}_{1}_{2} ({3}-{4})__'.format(right_now_time.day,
                                                  right_now_time.month,
                                                  right_now_time.year,
                                                  right_now_time.hour,
                                                  right_now_time.minute)
        return logdir


if __name__ == '__main__':
    LoadData.get_test_data()

    netw = AnsPredict(350)
    res_list = []

    for i in range(24):
        main_dataset = LoadData.load_ready_test_data(i)
        res_list.extend(list(netw.test_net(main_dataset)))

    arr_2_save = nmp.hstack(res_list).reshape(-1)

    print('total lines: ', len(arr_2_save))
    ready_csv = pnd.DataFrame(arr_2_save).to_csv('READY.csv')

    pass
