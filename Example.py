import numpy as nmp


class ClusterSL:
    """Класс образующий кластеры"""
    def __init__(self, left, right):
        """Инициализация кластера"""
        self.left_neighbour = left
        self.right_neighbour = right

    @staticmethod
    def line_generator(range_of=1000, num_of=10):
        """Генератор чисел для кластеризации"""
        new_line = nmp.random.randint(0, range_of, num_of)

        return new_line

    @staticmethod
    def go(new_line):
        sorted_line = nmp.sort(new_line)
        cl_line = list(sorted_line)
        len_betw = (sorted_line - nmp.roll(sorted_line, 1))[1:]
        seq = nmp.argsort(len_betw)

        for st_point, item in enumerate(seq):
            new_cluster = ClusterSL(cl_line[item], cl_line[item + 1])
            cl_line.pop(item + 1)
            cl_line[item] = new_cluster
            for num in range(st_point, len(seq)):
                if seq[num] > item:
                    seq[num] -= 1
            pass

        return cl_line[0]

    def count_elems(self):
        """Подсчёт элементов"""
        count = 0
        if type(self.left_neighbour) is ClusterSL:
            count += self.left_neighbour.count_elems()
        else:
            count += 1

        if type(self.right_neighbour) is ClusterSL:
            count += self.right_neighbour.count_elems()
        else:
            count += 1

        return count

    def count_levels(self, top_level=0):
        """Подсчёт уровней"""
        if type(self.left_neighbour) is ClusterSL:
            return self.left_neighbour.count_levels(top_level + 1)
        elif type(self.right_neighbour) is ClusterSL:
            return self.right_neighbour.count_levels(top_level + 1)
        else:
            return top_level

    def __repr__(self):
        return "[{}] / [{}]".format(self.left_neighbour, self.right_neighbour)

    def __int__(self):
        return (int(self.left_neighbour) + int(self.right_neighbour)) // 2


if __name__ == '__main__':
    test = ClusterSL.line_generator()
    result = ClusterSL.go(test)
