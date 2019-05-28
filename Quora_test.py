import numpy as nmp
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
import pandas as pnd
from matplotlib import pyplot as plt


class LoadData:
    @staticmethod
    def load_csv(path: str):
        datas = pnd.read_csv(path)
        return datas

    @staticmethod
    def prepare_corpus(loaded_csv: pnd.DataFrame):
        """Подготовка корпуса для обучения. Методы для эмбеддинга и для пар вопросов разные"""

        # Данные уже очищены, поэтому простое извлечение и конкатенация списка вопросов
        first_row = list(loaded_csv['question1'].dropna().values)
        second_row = list(loaded_csv['question2'].dropna().values)

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

            try:
                # Сохранение в виде списка индексов
                new_corpus.append([vocab[word] for word in sentence])
            except KeyError:
                # Если в словаре не обранужилось искомого слова
                loc_app = []

                # Для каждого слова
                for word in sentence:
                    # Если слово знакомое - сохраняем
                    if word in vocab:
                        loc_app.append(vocab[word])
                    else:
                        continue

                # Если в итоге вопрос остался пустым - заполняем заглушкой
                if len(loc_app) > 0:
                    new_corpus.append(loc_app)
                else:
                    loc_app.append(vocab['empty'])
                    new_corpus.append(loc_app)

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
    def get_test_data():
        """Метод для предварительной подготовки обучающего датасета"""
        print('Загрузка CSV')
        loaded_data = LoadData.load_csv('quora_not_net/clean_test.csv')
        print('Подготовка корпуса')
        first_q, second_q = LoadData.prepare_corpus(loaded_data)
        del loaded_data

        raw_dict = nmp.load('idx2w.npy')
        word2idx = {w: idx for (idx, w) in enumerate(raw_dict)}
        del raw_dict

        dense_vocab = LoadData.load_dense_vocab()

        # Так как файл с тестовыми вопросами оказался намного больше тренироваочного, пришлось разбивать датасет
        #   Наша цель просто прогнать обученную сеть по всем парам, поэтому вреда это не нанесёт
        slice_len = 100000
        for num, slice_range in enumerate(range(0, len(first_q), slice_len)):
            # Извлечение нужной части полного датасета
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
        # Загрузка нужной части разбитого датасета
        first_q_res = nmp.load('quora_test/q1_{}.npy'.format(part))
        second_q_res = nmp.load('quora_test/q2_{}.npy'.format(part))

        return first_q_res, second_q_res


class AnsPredict:
    """Нейронная сеть для сравнения вопросов"""
    def __init__(self, embedding_len):
        """
        Инициализация сети
        :param embedding_len: Длина эмбеддинга
        """

        # Плейсхолдеры для датасета
        self.input_ph_q1 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])
        self.input_ph_q2 = tf.placeholder(dtype=tf.float32, shape=[None, embedding_len])

        # Организация датасета и итератора из входных векторов
        dataset = tf.data.Dataset.from_tensor_slices((self.input_ph_q1, self.input_ph_q2)).cache()
        dataset = dataset.batch(50)
        self.iterate = dataset.make_initializable_iterator()
        input_batch_q1, input_batch_q2 = self.iterate.get_next()

        # В качестве источника информации для сети я взял матрицу, которая основана на пересечении данных двух векторов.
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

        # Полносвязный скрытый слой
        lay_2 = tf.layers.dense(inputs=lay_1_flat,
                                units=900,
                                activation=tf.nn.relu)

        # Выходной слой
        lay_3 = tf.layers.dense(inputs=lay_2,
                                units=1,
                                activation=tf.nn.sigmoid)

        # Выходное значение
        self.lay_3 = tf.reduce_mean(lay_3, 1)

    def test_net(self, loc_dataset, now_part, total_parts):
        """Метод для запуска процесса тестирования"""
        # Создание новой сессии
        sv = tf.Session()
        with sv as sess:
            # Инициализация
            sess.run(tf.global_variables_initializer())

            # Загрузка весов обученной сети
            saver = tf.train.Saver()
            saver.restore(sv, 'C:/TF/vk/save' + '/ckp')

            # Инициализация итератора
            sess.run(self.iterate.initializer, feed_dict={self.input_ph_q1: loc_dataset[0],
                                                          self.input_ph_q2: loc_dataset[1]})

            # Главный цикл. Определение числа проходов с учётом размеров блока на итерацию
            list_ans = []
            limit = int(nmp.ceil(len(loc_dataset[0]) / 50))
            gs = 0
            while gs < limit - 1:
                # Прогрессбар
                for _ in tqdm(range(limit),
                              total=limit,
                              ncols=100,
                              leave=False,
                              unit='b'):
                    # Получение вектора ответов
                    s = sess.run(self.lay_3)
                    list_ans.append(s)
                    gs += 1

        print("Готово: {} / {}".format(now_part + 1, total_parts))
        print('СКО в векторе ответов: ', nmp.hstack(list_ans).std())

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
    # Предварительная подготовка датасета и разбивка на части
    LoadData.get_test_data()

    # Инициализация сети
    embedding_len_param = 350
    netw = AnsPredict(embedding_len_param)

    # Цикл для подгрузки частей датасета
    res_list = []
    total = 24
    for i in range(total):
        main_dataset = LoadData.load_ready_test_data(i)
        res_list.extend(list(netw.test_net(main_dataset, i, total)))

    arr_2_save = nmp.hstack(res_list).reshape(-1)

    # Сохранение файла результатов
    print('total lines: ', len(arr_2_save))
    ready_csv = pnd.DataFrame(zip(nmp.arange(len(arr_2_save), dtype=nmp.int), arr_2_save),
                              columns=['test_id', 'is_duplicate']).set_index('test_id')
    ready_csv.to_csv('READY.csv')

    # Распредерение ответов, для интереса
    plt.hist(arr_2_save, 50)
    plt.show()

    pass
