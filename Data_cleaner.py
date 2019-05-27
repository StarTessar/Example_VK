import numpy as nmp
import pandas as pnd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords


class LoadData:
    @staticmethod
    def load_csv(path: str):
        datas = pnd.read_csv(path)
        return datas

    @staticmethod
    def save_csv(path: str, df_file: DataFrame):
        df_file.to_csv(path)


class MiningData:
    @staticmethod
    def count_repeat_ans(data: DataFrame):
        """Гистограмма распределения вопросов"""

        # Извлечение списка идентификаторов и построение гистограммы
        q_ids = nmp.hstack([data['qid1'].values, data['qid2'].values])
        hist = nmp.histogram(q_ids, nmp.max(q_ids))
        max_id = nmp.argmax(hist[0])
        plt.plot(hist[1][:-1], hist[0])
        plt.show()

        # Вывод самого частого вопроса и его пар
        qq = max_id + 1
        filt = len(data[data.qid1 == qq].question1) + len(data[data.qid2 == qq].question2)
        print(filt)
        filt_q = data[data.qid1 == qq][['question1', 'question2']]
        print(filt_q.head(10))
        pass

    @staticmethod
    def clean_corpus(data: DataFrame):
        """Очистка корпуса от лишних символов"""
        # Регулярное выражение для очистки корпуса от лишних символов и формирование списка стоп-слов
        reg = re.compile('[^a-z ]')
        loc_stopwords = [r'\b{}\b'.format(word.replace("'", '')) for word in stopwords.words('english')]
        print('Начало очистки')

        # Заполнение пустот
        print('  Заполнение пустот')
        clean_q1 = data['question1'].fillna('notastring')
        clean_q2 = data['question2'].fillna('notastring')

        # Приведение к нижнему регистру
        print('  Приведение к нижнему регистру')
        clean_q1 = clean_q1.str.lower()
        clean_q2 = clean_q2.str.lower()

        # Очистка от знаков
        print('  Очистка от знаков')
        clean_q1 = clean_q1.str.replace(reg, ' ')
        clean_q2 = clean_q2.str.replace(reg, ' ')

        # Удаление стоп-слов
        print('  Удаление стоп-слов')
        clean_q1 = clean_q1.str.replace(r'|'.join(loc_stopwords), ' ')
        clean_q2 = clean_q2.str.replace(r'|'.join(loc_stopwords), ' ')

        # Удачение лишних пробелов
        print('  Удаление лишних пробелов')
        clean_q1 = clean_q1.str.replace(r'\s+', ' ')
        clean_q2 = clean_q2.str.replace(r'\s+', ' ')
        clean_q1 = clean_q1.str.strip()
        clean_q2 = clean_q2.str.strip()

        # Заполнение новообразовавшихся пустот
        print('  Заполнение новообразовавшихся пустот')
        clean_q1 = clean_q1.fillna('notastring')
        clean_q2 = clean_q2.fillna('notastring')

        print('Очистка завершена!')
        data['question1'] = clean_q1
        data['question2'] = clean_q2

        nans_count = len(data[(data.question1 == 'notastring') | (data.question2 == 'notastring')].index)
        print('--Итого пустых значений: {0} / {1:.2%}'.format(nans_count, nans_count / data.shape[0]))

        return data

    @staticmethod
    def get_uniq_words(data: DataFrame):
        """Число уникальных слов в корпусе (т.е. в будущем словаре)"""
        # Исвлечение строк вопросов
        q_sent = list(data['question1'].values) + list(data['question2'].values)

        # Отбрасывание повторяющихся вопросов
        q_sent_uniq = set(q_sent)
        # Конкатенация строк в одну
        string_of_words = ' '.join(q_sent_uniq).lower()[1:]
        # Разбиение на слова
        set_of_words = sorted(set(string_of_words.split(' ')))
        print('Уникальных слов: {}'.format(len(set_of_words)))
        pass


if __name__ == '__main__':
    loaded = LoadData.load_csv('../test.csv')

    clean_loaded = MiningData.clean_corpus(loaded)
    LoadData.save_csv('clean_test.csv', clean_loaded)

    # MiningData.count_repeat_ans(clean_loaded)
    MiningData.get_uniq_words(loaded)
    pass
