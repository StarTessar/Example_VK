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
        """Подготовка корпуса для обучения"""

        # Данные уже очищены, поэтому простое извлечение и конкатенация списка вопросов
        first_row = list(loaded_csv['question1'].values)
        second_row = list(loaded_csv['question2'].values)

        result_list = first_row + second_row
        return result_list

    @staticmethod
    def tokenize_corpus(corpus):
        """Токенизация. Каждый вопрос в списке организуется в список составляющих слов"""
        tokens_list = [x.split() for x in corpus]
        return tokens_list

    @staticmethod
    def create_dict(corpus):
        """Создание словаря"""

        # Конкатенация в единую строку и развиение на отдельные слова. Затем отбрасывание повторений
        vocabulary = ' '.join(corpus).split(' ')
        vocabulary = set(vocabulary)

        # Построение прямого и обратного словарей
        word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

        # Подсчёт слов в словаре, для контроля
        vocabulary_size = len(vocabulary)
        print('Размер словаря: {}'.format(vocabulary_size))
        del vocabulary

        return word2idx, idx2word

    @staticmethod
    def create_dataset(tokenized_corpus, word2idx):
        """Создание датасета из корпуса"""
        window_size = 2     # Ширина окна для сопоставления контекста
        idx_pairs = []      # Инициализация списка для примеров

        # Для каждого вопроса в списке
        for sentence in tokenized_corpus:
            # Приведение предложения к численному виду в соответствии со словарём
            indices = [word2idx[word] for word in sentence]

            # Для каждого слова в вопросе
            for center_word_pos in range(len(indices)):
                # Внутри окна
                for w in range(-window_size, window_size + 1):
                    # Индекс контекстного слова
                    context_word_pos = center_word_pos + w
                    # Ограничение на краевые эффекты и самоповтор
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    # Выбор контекстного слова и сохранение в паре с текущим
                    context_word_idx = indices[context_word_pos]
                    idx_pairs.append((indices[center_word_pos], context_word_idx))

        idx_pairs = nmp.array(idx_pairs)
        return idx_pairs

    @staticmethod
    def prepare_dataset():
        """Метод для предварительной подготовки обучающего датасета"""
        print('Загрузка CSV')
        loaded_data = LoadData.load_csv('quora_not_net/clean_train.csv')
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
        """Загрузка предварительно подготовленного датасета"""
        print('Загрузка файлов CSV')
        raw_dict = nmp.load('idx2w.npy')
        raw_dataset = nmp.load('Ready_dataset.npy')

        # Перемешать не помешает
        nmp.random.shuffle(raw_dataset)

        # Восстановление словаря
        word2idx = {w: idx for (idx, w) in enumerate(raw_dict)}
        del raw_dict

        return raw_dataset, word2idx


class T2Wnet:
    """Нейронная сеть для эмбеддинга"""
    def __init__(self, vocab_size, embedding_len, learn_rate, batch_size):
        """
        Инициализация сети
        :param vocab_size: Размер словаря
        :param embedding_len: Длина эмбеддинга
        :param learn_rate: Параметр начальной скорости обучения
        :param batch_size: Размер блока данных для итерации
        """

        # Плейсхолдеры для датасета
        self.input_ph = tf.placeholder(dtype=tf.int32, shape=None)
        self.output_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        # Организация датасета и итератора из входных векторов
        dataset = tf.data.Dataset.from_tensor_slices((self.input_ph, self.output_ph)).cache()
        dataset = dataset.repeat().batch(batch_size)
        self.iterate = dataset.make_initializable_iterator()
        input_batch, output_batch = self.iterate.get_next()

        # Создание слоя для эмбеддинга. В него будет смотреть сеть для воспроизведения слов в векторной форме
        self.embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_len], -1.0, 1.0))

        # Организация второго слоя
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocab_size, embedding_len],
                                stddev=1.0 / nmp.math.sqrt(embedding_len)))
        softmax_biases = tf.Variable(tf.zeros([vocab_size]))

        # Векторизация числового представления слов
        embed = tf.nn.embedding_lookup(self.embeddings, input_batch)
        # Расчёт функции потерь
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                       labels=output_batch, num_sampled=64, num_classes=vocab_size))
        tf.summary.scalar("loss", loss)

        # Глобальный счётчик итераций
        with tf.variable_scope("gs"):
            self.global_step = tf.train.get_or_create_global_step()

        # Вычисление текущего значения скорости обучения. Для улучшения сходимости сделал её ниспадающей со временем
        self.loc_lear = learn_rate * tf.pow(tf.cast(0.1, tf.float64), self.global_step / 4000 + 1)
        tf.summary.scalar("learning_rate", self.loc_lear)
        # Оптимизация весов
        self.optim = tf.train.AdamOptimizer(learning_rate=self.loc_lear).minimize(loss,
                                                                                  global_step=self.global_step)

    def teach(self, limit, loc_dataset_raw):
        """Метод для запуска процесса обучения"""
        # Создание новой сессии
        sv = tf.Session()
        with sv as sess:
            # Инициализация
            summ = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            # Перемешивание датасета и инициализация итератора
            loc_dataset = loc_dataset_raw
            nmp.random.shuffle(loc_dataset)
            sess.run(self.iterate.initializer, feed_dict={self.input_ph: loc_dataset[:, 1],
                                                          self.output_ph: loc_dataset[:, 0, nmp.newaxis]})

            # Подготовка логирования
            writer = tf.summary.FileWriter('C:/TF/Log/' + 'Vars' + self.get_log_directory())
            writer.add_graph(sess.graph)

            # Главный обучающий цикл
            gs = 0
            while gs < limit - 1:
                # Для удобства прикрутил прогрессбар
                for _ in tqdm(range(limit),
                              total=limit,
                              ncols=100,
                              leave=False,
                              unit='b'):

                    # Выполнение итерации
                    gs_loc, _ = sess.run([self.global_step, self.optim])

                    # Логирование
                    s = sess.run(summ)
                    writer.add_summary(s, gs)

                    gs += 1

                    # Для удаления цикличности данных они перемешиваются после каждой эпохи
                    if gs % (len(dataset_val) // batch_size_param) == 0:
                        nmp.random.shuffle(loc_dataset)
                        sess.run(self.iterate.initializer, feed_dict={self.input_ph: loc_dataset[:, 1],
                                                                      self.output_ph: loc_dataset[:, 0, nmp.newaxis]})

            # Последняя запись в лог и сохранение весов из скрытого слоя эмбеддинга
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
    # Предварительная подготовка датасета
    LoadData.prepare_dataset()

    # Загрузка соранённого датасета
    dataset_val, word2idx_dict = LoadData.load_prepared_datas()

    # Установка параметров для обучения. Размер блока для итерации, число эпох и длина эмбеддинга
    voc_size = len(word2idx_dict)
    batch_size_param = 2000
    epo_train_param = 5
    embedding_len_param = 350
    learn_rate_param = 1E-2

    # Инициализация сети
    t2w_net = T2Wnet(voc_size, embedding_len_param, learn_rate_param, batch_size_param)
    # Запуск обучения
    wei = t2w_net.teach(epo_train_param * len(dataset_val) // batch_size_param, dataset_val)

    # Сохранение словаря эмбеддингов
    nmp.save('Result_embeddings', wei)
    pass
