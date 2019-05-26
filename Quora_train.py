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
            new_corpus.append([vocab[word] for word in sentence if word is not 'nan'])

        return new_corpus

    @staticmethod
    def load_dense_vocab():
        raw_weis = nmp.load('Result_embeddings.npy')
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
    def get_train_data():
        print('Загрузка CSV')
        loaded_data = LoadData.load_csv('train.csv')
        print('Подготовка корпуса')
        first_q, second_q = LoadData.prepare_corpus(loaded_data)
        print('Токенизация')
        first_q_token = LoadData.tokenize_corpus(first_q)
        second_q_token = LoadData.tokenize_corpus(second_q)

        raw_dict = nmp.load('idx2w.npy')
        word2idx = {w: idx for (idx, w) in enumerate(raw_dict)}

        del raw_dict
        print('Векторизация')
        first_q_vect = LoadData.vector_corpus(first_q_token, word2idx)
        second_q_vect = LoadData.vector_corpus(second_q_token, word2idx)
        dub_arr = loaded_data['is_duplicate'].values

        dense_vocab = LoadData.load_dense_vocab()
        first_q_res = LoadData.vector_sent(first_q_vect, dense_vocab)
        second_q_res = LoadData.vector_sent(second_q_vect, dense_vocab)

        print('Сохранение данных')
        nmp.save('quora_train_data/q1', first_q_res)
        nmp.save('quora_train_data/q2', second_q_res)
        nmp.save('quora_train_data/ans', dub_arr)

    @staticmethod
    def load_ready_train_data():
        first_q_res = nmp.load('quora_train_data/q1.npy')
        second_q_res = nmp.load('quora_train_data/q2.npy')
        dub_arr = nmp.load('quora_train_data/ans.npy')

        norm_param_mean = nmp.vstack([first_q_res, second_q_res]).mean(axis=0)

        first_q_res = (first_q_res - norm_param_mean)
        second_q_res = (second_q_res - norm_param_mean)

        norm_param_std = nmp.vstack([first_q_res, second_q_res]).std(axis=0)

        first_q_res = (first_q_res / norm_param_std)
        second_q_res = (second_q_res / norm_param_std)

        rnd_state = nmp.random.get_state()
        nmp.random.shuffle(first_q_res)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(second_q_res)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(dub_arr)

        return first_q_res, second_q_res, dub_arr


class AnsPredict:
    def __init__(self, embedding_len, learn_rate, batch_size):
        self.input_ph_q1 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])
        self.input_ph_q2 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])
        self.output_ph = tf.placeholder(dtype=tf.int32, shape=[None])

        dataset = tf.data.Dataset.from_tensor_slices((self.input_ph_q1, self.input_ph_q2, self.output_ph)).cache()
        dataset = dataset.repeat().batch(batch_size)
        self.iterate = dataset.make_initializable_iterator()
        input_batch_q1, input_batch_q2, output_batch = self.iterate.get_next()

        len_q1 = tf.sqrt(tf.reduce_sum(tf.square(input_batch_q1)))
        len_q2 = tf.sqrt(tf.reduce_sum(tf.square(input_batch_q2)))
        cos_dist = tf.reduce_sum(input_batch_q1 * input_batch_q2) / (len_q1 * len_q2)

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

        # attention_plane = tf.matmul(lay_q1[:, :, tf.newaxis], lay_q2[:, :, tf.newaxis],
        #                             adjoint_b=True)
        # attention_plane = attention_plane[:, :, :, tf.newaxis]
        #
        # lay_1 = tf.layers.conv2d(inputs=attention_plane,
        #                          filters=5,
        #                          kernel_size=[5, 5],
        #                          strides=[5, 5],
        #                          activation=None)

        # lay_1 = tf.layers.conv2d(inputs=lay_1,
        #                          filters=10,
        #                          kernel_size=[5, 5],
        #                          strides=[1, 1],
        #                          activation=tf.nn.relu)

        # lay_1_flat = tf.layers.flatten(lay_1)

        lay_q_flat = tf.layers.dropout(lay_q, 0.1)

        lay_1_flat = lay_q_flat

        lay_2 = tf.layers.dense(inputs=lay_1_flat,
                                units=3000,
                                activation=tf.nn.relu)

        lay_3 = tf.layers.dense(inputs=lay_2,
                                units=1,
                                activation=tf.nn.sigmoid)

        lay_3 = tf.reduce_mean(lay_3, 1)
        tf.summary.histogram("mean_ans", lay_3)

        all_mean = tf.reduce_max(2 / tf.exp(lay_3 - tf.reduce_mean(lay_3)))
        tf.summary.histogram("all_simult", all_mean)

        with tf.variable_scope("gs"):
            self.global_step = tf.train.get_or_create_global_step()

        self.xent = tf.reduce_mean(tf.losses.log_loss(labels=output_batch, predictions=lay_3)) * all_mean
        tf.summary.scalar("xent_1", self.xent)

        logloss = tf.reduce_max(tf.losses.log_loss(labels=output_batch, predictions=lay_3))
        tf.summary.scalar("logloss", logloss)

        self.correct_prediction = tf.abs(lay_3 - tf.cast(output_batch, tf.float32))
        self.accuracy = 1 - tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)

        self.loc_lear = learn_rate * tf.pow(tf.cast(0.1, tf.float64), self.global_step / 500 + 1)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.loc_lear).minimize(self.xent,
                                                                                  global_step=self.global_step)
        tf.summary.scalar("learning_rate", self.loc_lear)

    def teach(self, epo, loc_dataset):
        sv = tf.Session()
        with sv as sess:
            summ = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            sess.run(self.iterate.initializer, feed_dict={self.input_ph_q1: loc_dataset[0],
                                                          self.input_ph_q2: loc_dataset[1],
                                                          self.output_ph: loc_dataset[2]})

            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('C:/TF/Log/' + 'quora' + self.get_log_directory())
            writer.add_graph(sess.graph)

            limit = epo
            gs = 0
            while gs < limit - 1:
                for _ in tqdm(range(limit),
                              total=limit,
                              ncols=100,
                              leave=False,
                              unit='b'):
                    # for data, target in loc_dataset:
                    gs_loc, _ = sess.run([self.global_step, self.optim])

                    s = sess.run(summ)
                    writer.add_summary(s, gs)

                    gs += 1

            s = sess.run(summ)
            writer.add_summary(s, gs)
            saver.save(sess, 'C:/TF/vk/save' + '/ckp')

        print("Готово")

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
    LoadData.get_train_data()

    batch_size_of = 10000
    main_dataset = LoadData.load_ready_train_data()
    netw = AnsPredict(main_dataset[0].shape[1], 1E-2, batch_size_of)
    netw.teach(1000, main_dataset)
    # netw.teach(len(main_dataset[0]) // batch_size_of, main_dataset)
    pass
