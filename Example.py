import numpy as nmp
from matplotlib import pyplot as plt


class ClusterSL:
    """Класс образующий кластеры"""
    def __init__(self, left, right):
        """Инициализация кластера"""
        self.left_neighbour = left
        self.right_neighbour = right

    @staticmethod
    def line_int_generator(range_of=1000, num_of=10):
        """Генератор чисел для кластеризации"""
        new_line = nmp.random.randint(0, range_of, num_of)

        return new_line

    @staticmethod
    def line_fl_generator(range_of=1000, num_of=10):
        """Генератор чисел для кластеризации"""
        new_line = nmp.random.rand(num_of) * range_of

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

    def draw_branch(self, img, top_level=0):
        """Отрисовка ветви"""
        # line_width = img.shape[1] // 400
        # line_h = img.shape[0] // 200
        line_width = 1
        line_h = 1

        level_height = 50
        branch_pos = int(self)
        branch_lev = img.shape[0] - (self.count_levels() + 1) * level_height - 1
        img[branch_lev - line_h:branch_lev + line_h, int(self.left_neighbour):int(self.right_neighbour)] = 1
        img[top_level:branch_lev, branch_pos - line_width:branch_pos + line_width] = 1

        # plt.imshow(img)
        # plt.gray()
        # plt.show()

        if type(self.left_neighbour) is ClusterSL:
            self.left_neighbour.draw_branch(img, branch_lev)
        else:
            img[branch_lev:, int(self.left_neighbour) - line_width:int(self.left_neighbour) + line_width] = 1

        if type(self.right_neighbour) is ClusterSL:
            self.right_neighbour.draw_branch(img, branch_lev)
        else:
            img[branch_lev:, int(self.right_neighbour) - line_width:int(self.right_neighbour) + line_width] = 1

    def draw_tree(self):
        """Отрисовка дерева"""
        plate_img = self.prep_img()
        self.draw_branch(plate_img)

        return plate_img

    def prep_img(self):
        """Подготовка полотна"""
        level_height = 50
        tree_h = self.count_levels() + 2
        tree_w = int(self.max_val())
        tree_shift = int(self.min_val())
        new_img = nmp.zeros([tree_h * level_height, tree_w + tree_shift])

        return new_img

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
        br_1 = top_level
        br_2 = top_level
        if type(self.left_neighbour) is ClusterSL:
            br_1 = self.left_neighbour.count_levels(top_level + 1)
        if type(self.right_neighbour) is ClusterSL:
            br_2 = self.right_neighbour.count_levels(top_level + 1)

        return max(br_1, br_2)

    def max_val(self):
        """Максимум"""
        if type(self.right_neighbour) is ClusterSL:
            return self.right_neighbour.max_val()
        else:
            return self.right_neighbour

    def min_val(self):
        """Минимум"""
        if type(self.left_neighbour) is ClusterSL:
            return self.left_neighbour.min_val()
        else:
            return self.left_neighbour

    def __gt__(self, other):
        return self.right_neighbour > other

    def __lt__(self, other):
        return self.left_neighbour < other

    def __repr__(self):
        return "[{}] - [{}]".format(self.left_neighbour, self.right_neighbour)

    def __int__(self):
        return (int(self.left_neighbour) + int(self.right_neighbour)) // 2


def simple_test():
    test = ClusterSL.line_int_generator(num_of=10)
    result = ClusterSL.go(test)
    img_res = result.draw_tree()

    plt.imshow(img_res)
    plt.gray()
    plt.show()


def multi_test():
    for i in range(10):
        test = ClusterSL.line_fl_generator(num_of=100, range_of=1000)
        result = ClusterSL.go(test)
        img_res = result.draw_tree()

        plt.imshow(img_res)
        plt.gray()
        # plt.savefig(str(i + 1) + '.png')
        plt.imsave(str(i + 1) + '.png', img_res)


if __name__ == '__main__':
    multi_test()
    pass
