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

        result_list = first_row + second_row
        return result_list

    @staticmethod
    def tokenize_corpus(corpus):
        tokens_list = [x.split() for x in corpus]
        return tokens_list

    @staticmethod
    def create_dict(corpus):
        vocabulary = ' '.join(corpus).split(' ')
        vocabulary = set(vocabulary)
        vocabulary.remove('')

        word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

        vocabulary_size = len(vocabulary)
        print('Размер словаря: {}'.format(vocabulary_size))
        del vocabulary

        return word2idx, idx2word

    @staticmethod
    def create_dict_old(tokenized_corpus):
        vocabulary = []
        for sentence in tokenized_corpus:
            for token in sentence:
                if token not in vocabulary:
                    vocabulary.append(token)

        word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

        vocabulary_size = len(vocabulary)
        print('Размер словаря: {}'.format(vocabulary_size))

        return word2idx, idx2word

    @staticmethod
    def create_dataset(tokenized_corpus, word2idx):
        window_size = 2
        idx_pairs = []

        for sentence in tokenized_corpus:
            indices = [word2idx[word] for word in sentence]

            for center_word_pos in range(len(indices)):
                for w in range(-window_size, window_size + 1):
                    context_word_pos = center_word_pos + w
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))

        idx_pairs = nmp.array(idx_pairs)

        return idx_pairs

    @staticmethod
    def create_dataset_old(tokenized_corpus, word2idx):
        window_size = 2
        idx_pairs = []

        for sentence in tokenized_corpus:
            indices = [word2idx[word] for word in sentence]

            for center_word_pos in range(len(indices)):
                for w in range(-window_size, window_size + 1):
                    context_word_pos = center_word_pos + w
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))

        idx_pairs = nmp.array(idx_pairs)

        return idx_pairs

    @staticmethod
    def prepare_dataset():
        print('Загрузка CSV')
        loaded_data = LoadData.load_csv('train.csv')
        print('Подготовка корпуса')
        ready_corpus = LoadData.prepare_corpus(loaded_data)

        print('Токенизация')
        tokens = LoadData.tokenize_corpus(ready_corpus)
        print('Создание словарей')
        word2idx_dict, idx2word_dict = LoadData.create_dict(ready_corpus)
        print('Подготовка датасета')
        dataset = LoadData.create_dataset(tokens, word2idx_dict)

        print('Сохранение датасета')
        dataset = nmp.unique(dataset, axis=0)
        nmp.save('Ready_dataset.npy', dataset)
        
        print('Сохранение idx2w словаря')
        nmp.save('idx2w.npy', list(word2idx_dict))

    @staticmethod
    def load_prepared_datas():
        print('Загрузка файлов CSV')
        raw_dict = nmp.load('idx2w.npy')
        raw_dataset = nmp.load('Ready_dataset.npy')
        nmp.random.shuffle(raw_dataset)

        word2idx = {w: idx for (idx, w) in enumerate(raw_dict)}
        del raw_dict

        return raw_dataset, word2idx


class T2Wnet:
    def __init__(self, vocab_size, embedding_len, learn_rate, batch_size):
        self.input_ph = tf.placeholder(dtype=tf.int32, shape=None)
        self.output_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        dataset = tf.data.Dataset.from_tensor_slices((self.input_ph, self.output_ph)).cache()
        dataset = dataset.repeat().batch(batch_size)
        self.iterate = dataset.make_initializable_iterator()
        input_batch, output_batch = self.iterate.get_next()

        self.embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_len], -1.0, 1.0))
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocab_size, embedding_len],
                                stddev=1.0 / nmp.math.sqrt(embedding_len)))
        softmax_biases = tf.Variable(tf.zeros([vocab_size]))

        embed = tf.nn.embedding_lookup(self.embeddings, input_batch)
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                       labels=output_batch, num_sampled=64, num_classes=vocab_size))

        with tf.variable_scope("gs"):
            self.global_step = tf.train.get_or_create_global_step()

        tf.summary.scalar("xent_1", loss)

        self.loc_lear = learn_rate * tf.pow(tf.cast(0.1, tf.float64), self.global_step / 4000 + 1)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.loc_lear).minimize(loss,
                                                                                  global_step=self.global_step)
        tf.summary.scalar("learning_rate", self.loc_lear)

    def teach(self, epo, loc_dataset):
        sv = tf.Session()
        with sv as sess:
            summ = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            sess.run(self.iterate.initializer, feed_dict={self.input_ph: loc_dataset[:, 0],
                                                          self.output_ph: loc_dataset[:, 1, nmp.newaxis]})

            writer = tf.summary.FileWriter('C:/TF/Log/' + 'Vars' + self.get_log_directory())
            writer.add_graph(sess.graph)

            limit = epo
            gs = 0
            while gs < limit - 1:
                for _ in tqdm(range(limit),
                              total=limit,
                              ncols=100,
                              leave=False,
                              unit='b'):
                    gs_loc, _ = sess.run([self.global_step, self.optim])

                    s = sess.run(summ)
                    writer.add_summary(s, gs)

                    gs += 1

            s = sess.run(summ)
            writer.add_summary(s, gs)

            emb = sess.run(self.embeddings)
        print("Готово")
        return emb

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
    dataset_val, word2idx_dict = LoadData.load_prepared_datas()

    voc_size = len(word2idx_dict)
    batch_size_of = 2000
    t2w_net = T2Wnet(voc_size, 350, 1E-2, batch_size_of)
    wei = t2w_net.teach(len(dataset_val) // batch_size_of, dataset_val)

    nmp.save('Result_embeddings', wei)
    pass
