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
        """Подготовка корпуса для обучения. Методы для эмбеддинга и для пар вопросов разные"""

        # Данные уже очищены, поэтому простое извлечение и конкатенация списка вопросов
        first_row = list(loaded_csv['question1'].fillna('notastring').values)
        second_row = list(loaded_csv['question2'].fillna('notastring').values)

        # Возвращает два списка вопросов
        return first_row, second_row

    @staticmethod
    def tokenize_corpus(corpus):
        """Токенизация. Каждый вопрос в списке организуется в список составляющих слов"""
        tokens_list = [x.split() for x in corpus]
        return tokens_list

    @staticmethod
    def vector_corpus(tokenized_corpus, vocab):
        """Представление слов в виде индексов словаря"""
        new_corpus = []

        # Для каждого вопроса
        for sentence in tokenized_corpus:
            # Контроль исключений
            while 'nan' in sentence:
                sentence.remove('nan')
            while 'null' in sentence:
                sentence.remove('null')
            if len(sentence) == 0:
                sentence.append('empty')
                # Сохранение в виде списка индексов
            new_corpus.append([vocab[word] for word in sentence if word is not 'nan'])

        return new_corpus

    @staticmethod
    def load_dense_vocab():
        """Загрузка словаря эмбеддингов"""
        raw_weis = nmp.load('Result_embeddings.npy')
        # Контроль ориентации измерений
        if raw_weis.shape[1] > 1000:
            raw_weis = nmp.rot90(raw_weis)
        return raw_weis

    @staticmethod
    def vector_sent(arr_of_wordvect: list, dense_vocab: dict):
        """Векторизация вопроса в соответствии с эмбеддингами"""
        new_array = []

        # Для каждого вопроса
        for line in arr_of_wordvect:
            # Список слов в виде сжатых векторов
            word_vectors = [dense_vocab[word] for word in line]
            # Суммирование векторных представлений слов в вопросе.
            #   В моём понимании это должно давать сумму их семантических значений, что является смыслом предложения
            #   Этот подход не учитывает последовательность слов, которая бывает критична, но зато прост в применении
            line_vector = sum(word_vectors)
            new_array.append(line_vector)

        return new_array

    @staticmethod
    def get_train_data():
        """Метод для предварительной подготовки обучающего датасета"""
        print('Загрузка CSV')
        loaded_data = LoadData.load_csv('quora_not_net/clean_train.csv')
        print('Подготовка корпуса')
        first_q, second_q = LoadData.prepare_corpus(loaded_data)
        print('Токенизация')
        first_q_token = LoadData.tokenize_corpus(first_q)
        second_q_token = LoadData.tokenize_corpus(second_q)

        # Восстановление словаря
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
        """Загрузка предварительно подготовленного датасета"""
        first_q_res = nmp.load('quora_train_data/q1.npy')
        second_q_res = nmp.load('quora_train_data/q2.npy')
        dub_arr = nmp.load('quora_train_data/ans.npy')

        # Нормализация данных по каждому измерению, чтобы сети было проще работать
        norm_param_mean = nmp.vstack([first_q_res, second_q_res]).mean(axis=0)
        first_q_res = (first_q_res - norm_param_mean)
        second_q_res = (second_q_res - norm_param_mean)

        norm_param_std = nmp.vstack([first_q_res, second_q_res]).std(axis=0)
        first_q_res = (first_q_res / norm_param_std)
        second_q_res = (second_q_res / norm_param_std)

        # Перемешивание датасета
        rnd_state = nmp.random.get_state()
        nmp.random.shuffle(first_q_res)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(second_q_res)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(dub_arr)

        return first_q_res, second_q_res, dub_arr


class AnsPredict:
    """Нейронная сеть для сравнения вопросов"""
    def __init__(self, embedding_len, learn_rate, batch_size):
        """
        Инициализация сети
        :param embedding_len: Длина эмбеддинга
        :param learn_rate: Параметр начальной скорости обучения
        :param batch_size: Размер блока данных для итерации
        """

        # Плейсхолдеры для датасета
        self.input_ph_q1 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])
        self.input_ph_q2 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])
        self.output_ph = tf.placeholder(dtype=tf.int32, shape=[None])

        # Организация датасета и итератора из входных векторов
        dataset = tf.data.Dataset.from_tensor_slices((self.input_ph_q1, self.input_ph_q2, self.output_ph)).cache()
        dataset = dataset.repeat().batch(batch_size)
        self.iterate = dataset.make_initializable_iterator()
        input_batch_q1, input_batch_q2, output_batch = self.iterate.get_next()

        # В качестве источника информации для сети я взял матрицу, которая основана на пересечении данных двух векторов.
        #   Для человека это не будет иметь никакого смысла,
        #   но я предположил, что сеть сможет обнаружить закономерности.
        attention_plane = tf.matmul(input_batch_q1[:, :, tf.newaxis], input_batch_q2[:, :, tf.newaxis],
                                    adjoint_b=True)
        attention_plane = attention_plane[:, :, :, tf.newaxis]

        # Для извлечения карт признаков определён свёрточный слой
        lay_1 = tf.layers.conv2d(inputs=attention_plane,
                                 filters=5,
                                 kernel_size=[5, 5],
                                 strides=[5, 5],
                                 activation=None)

        # Развёртывание для полносвязного слоя
        lay_1_flat = tf.layers.flatten(lay_1)

        # Дропаут для устойчивости к шумам
        lay_1_flat = tf.layers.dropout(lay_1_flat, 0.1)

        # Полносвязный скрытый слой
        lay_2 = tf.layers.dense(inputs=lay_1_flat,
                                units=900,
                                activation=tf.nn.relu)

        # Выходной слой
        lay_3 = tf.layers.dense(inputs=lay_2,
                                units=1,
                                activation=tf.nn.sigmoid)

        # Контроль распределения ответов в логе
        lay_3 = tf.reduce_mean(lay_3, 1)
        tf.summary.histogram("distr_ans", lay_3)

        # Ввод контроля распределения ответов. Чем меньше разброс вокруг среднего, тем больше значение
        # all_mean = tf.reduce_max(2 / tf.exp(lay_3 - tf.reduce_mean(lay_3)))
        all_mean = tf.reduce_max(2 / tf.exp(tf.math.reduce_std(lay_3)))
        tf.summary.histogram("all_simult", all_mean)

        # Глобальный счётчик итераций
        with tf.variable_scope("gs"):
            self.global_step = tf.train.get_or_create_global_step()

        # Расчёт функции потерь. Здесь, помимо потерь самих ответов сеть наказывается и за поиск одного среднего ответа
        #   Удовлетворяющего заданному условию
        self.loss_func = tf.reduce_mean(tf.losses.log_loss(labels=output_batch, predictions=lay_3)) * all_mean
        tf.summary.scalar("loss_func", self.loss_func)

        # Расчёт метрики
        self.logloss = tf.reduce_max(tf.losses.log_loss(labels=output_batch, predictions=lay_3))
        tf.summary.scalar("logloss", self.logloss)

        # Вычисление текущего значения скорости обучения
        self.loc_lear = learn_rate * tf.pow(tf.cast(0.1, tf.float64), self.global_step / 500 + 1)
        tf.summary.scalar("learning_rate", self.loc_lear)
        # Оптимизация весов
        self.optim = tf.train.AdamOptimizer(learning_rate=self.loc_lear).minimize(self.loss_func,
                                                                                  global_step=self.global_step)

    @staticmethod
    def shuffle_and_split(loc_dataset, test_scale):
        """Перемешивание датасета и разбиение на тестовую и тренировочную выборки"""
        ph_q1: nmp.ndarray = loc_dataset[0]
        ph_q2: nmp.ndarray = loc_dataset[1]
        ph_out: nmp.ndarray = loc_dataset[2]
        
        # Перемешивание
        rnd_state = nmp.random.get_state()
        nmp.random.shuffle(ph_q1)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(ph_q2)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(ph_out)
        
        # Выборка для теста
        split_index = int(len(loc_dataset[0]) * test_scale)
        ph_q1_test = ph_q1[:split_index]
        ph_q2_test = ph_q2[:split_index]
        ph_out_test = ph_out[:split_index]
        test_dataset = [ph_q1_test, ph_q2_test, ph_out_test]
        
        # Выборка для обучения
        ph_q1_train = ph_q1[split_index:]
        ph_q2_train = ph_q2[split_index:]
        ph_out_train = ph_out[split_index:]
        train_dataset = [ph_q1_train, ph_q2_train, ph_out_train]

        return test_dataset, train_dataset

    @staticmethod
    def just_shuffle(loc_dataset):
        """Перемешивание датасета"""
        ph_q1: nmp.ndarray = loc_dataset[0]
        ph_q2: nmp.ndarray = loc_dataset[1]
        ph_out: nmp.ndarray = loc_dataset[2]
        
        # Перемешиание
        rnd_state = nmp.random.get_state()
        nmp.random.shuffle(ph_q1)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(ph_q2)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(ph_out)

        return ph_q1, ph_q2, ph_out

    def teach(self, limit, loc_dataset):
        """Метод для запуска процесса обучения"""
        # Создание новой сессии
        sv = tf.Session()
        with sv as sess:
            # Инициализация
            summ = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            # Перемешивание датасета и инициализация итератора, организация тестовой выборки
            test_scale = 0.2

            test_dataset, train_dataset = AnsPredict.shuffle_and_split(loc_dataset, test_scale)
            ph_q1_test, ph_q2_test, ph_out_test = test_dataset
            ph_q1_train, ph_q2_train, ph_out_train = train_dataset

            sess.run(self.iterate.initializer, feed_dict={self.input_ph_q1: ph_q1_train,
                                                          self.input_ph_q2: ph_q2_train,
                                                          self.output_ph: ph_out_train})

            # Подготовка логирования и сохранения сети
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('C:/TF/Log/' + 'quora' + self.get_log_directory())
            test_writer = tf.summary.FileWriter('C:/TF/Log/' + 'quora_valid' + self.get_log_directory())
            writer.add_graph(sess.graph)

            # Главный обучающий цикл
            gs = 0
            while gs < limit - 1:
                # Прогрессбар
                for _ in tqdm(range(limit),
                              total=limit,
                              ncols=100,
                              leave=False,
                              unit='b'):

                    # Выполнение итерации
                    gs_loc, _ = sess.run([self.global_step, self.optim])

                    if gs % 100 == 0:
                        # Логирование
                        s = sess.run(summ)
                        writer.add_summary(s, gs)

                    gs += 1

                    # Для удаления цикличности данных они перемешиваются после каждой эпохи
                    if (gs % (len(loc_dataset[0]) // batch_size_param) == 0) | (gs == 1):
                        # Тестовый прогон
                        sess.run(self.iterate.initializer, feed_dict={self.input_ph_q1: ph_q1_test,
                                                                      self.input_ph_q2: ph_q2_test,
                                                                      self.output_ph: ph_out_test})
                        test_summary = []
                        for test_iter in range(len(test_dataset[0]) // batch_size_param):
                            test_summary.extend(sess.run([self.logloss]))
                        test_summary = tf.Summary(
                            value=[tf.Summary.Value(
                                tag='logloss', simple_value=nmp.hstack(test_summary).mean())])
                        test_writer.add_summary(test_summary, gs)

                        # Загрузка обучающей выборки
                        ph_q1_train, ph_q2_train, ph_out_train = AnsPredict.just_shuffle(train_dataset)
                        sess.run(self.iterate.initializer, feed_dict={self.input_ph_q1: ph_q1_train,
                                                                      self.input_ph_q2: ph_q2_train,
                                                                      self.output_ph: ph_out_train})

            # Последняя запись в лог и сохранение весов обученной сети
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
    # Предварительная подготовка датасета
    LoadData.get_train_data()

    # Загрузка соранённого датасета
    main_dataset = LoadData.load_ready_train_data()

    # Установка параметров для обучения. Размер блока для итерации и число эпох
    batch_size_param = 100
    epo_train_param = 5
    learn_rate_param = 1E-2

    # Инициализация сети
    netw = AnsPredict(main_dataset[0].shape[1], learn_rate_param, batch_size_param)
    # Запуск обучения
    netw.teach(epo_train_param * len(main_dataset[0]) // batch_size_param, main_dataset)
    pass
