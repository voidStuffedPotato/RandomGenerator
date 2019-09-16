from decimal import Decimal, getcontext
from statistics import mean, variance, stdev
from numpy.random import randint, uniform, normal
from itertools import accumulate
from math import sqrt


getcontext().prec = 4


class Style:
    ITALIC = '\033[3m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Randomizer:
    def __init__(self, name):
        self.name = name
        self.sample = self.get_randoms()
        self.mean = mean(self.sample)
        self.variance = variance(self.sample)
        self.stdev = stdev(self.sample)

    def get_randoms(self):
        raise NotImplementedError

    def __str__(self):
        name = self.name
        mean = 'Среднее выборочное: %.4f' % self.mean
        var = 'Дисперсия: %.4f' % self.variance
        std = 'Среднеквадратичное отклонение: %.4f' % self.stdev

        return '\n'.join((name, '', mean, var, std, ''))


class TableMethod(Randomizer):
    def __init__(self):
        super().__init__(Style.ITALIC + 'Табличный метод' + Style.END)

    def get_randoms(self):
        """
        Возвращает список случайных чисел из таблицы случайных чисел
        """
        randoms = list()
        with open('rand_table.txt', 'r') as f:
            for line in f:
                line = line.split()
                randoms.extend(map(lambda x: Decimal(x)/Decimal(100), line))
        return randoms


class LCG(Randomizer):
    def __init__(self, a=3, c=2, m=2239, seed=24, iterations=50000):
        self.a = a
        self.c = c
        self.m = m
        self.seed = seed
        self.iterations = iterations
        super().__init__(Style.ITALIC + f'Линейный конгруэнтный метод (m={m})' + Style.END)

    def get_randoms(self):
        a = self.a
        c = self.c
        m = self.m

        def generate(seed):
            x = seed
            yield x
            for i in range(self.iterations-1):
                x = (a * x + c) % m
                yield x
        lst = list(generate(self.seed))
        return lst


class UniformInt(Randomizer):
    def __init__(self, low, high, iterations=50000):
        self.low = low
        self.high = high
        self.iterations = iterations
        super().__init__(Style.ITALIC + f'Дискретное равномерное распределенеие (a={low}, b={high - 1})' + Style.END)

    def get_randoms(self):
        return list(randint(self.low, self.high, self.iterations))


class UniformFloat(Randomizer):
    def __init__(self, low, high, iterations=1000):
        self.low = low
        self.high = high
        self.iterations = iterations
        super().__init__(Style.ITALIC + f'Равномерное распределенеие (a={low}, b={high})' + Style.END)

    def get_randoms(self):
        return list(uniform(self.low, self.high, self.iterations))


class Neumann(Randomizer):
    def __init__(self, low, high, iterations=50000):
        self.low = low
        self.high = high
        self.iterations = iterations
        self.C = (12 + (high - low) ** 3) / (12 * (high - low))
        self.m = (low + high) / 2
        super().__init__(Style.ITALIC + f'Метод Неймана (a = {low}, b = {high})' + Style.END)

    def get_randoms(self):

        def dist_func(y):
            return self.C - (y - self.m) ** 2

        def generate():
            r_1 = uniform(0, 1)
            r_2 = uniform(0, 1)
            alpha = self.low + r_1 * (self.high - self.low)
            beta = r_2 * self.C
            return alpha if beta < dist_func(alpha) else None

        arr = []
        while True:
            gen = generate()
            if gen is not None:
                arr.append(gen)
            if len(arr) == self.iterations:
                break
        return arr


class CLT(Randomizer):
    def __init__(self, mean, variance, iterations=50000):
        self.iterations = iterations
        super().__init__(Style.ITALIC + f'Центральная предельная теорема' + Style.END)
        self.mean_ = mean
        self.variance_ = variance
        self.transform()

    def get_randoms(self):
        lst = list(accumulate(uniform(0, 1, self.iterations)))
        lst = [Decimal(i) for i in lst]
        return lst

    def transform(self):
        lst = [(i - self.mean)/self.variance for i in self.sample]
        # lst = [(i - Decimal(self.iterations/2)) / Decimal(sqrt(self.iterations/12)) for i in self.sample]
        lst = [i * Decimal(sqrt(self.variance_)) + self.mean_ for i in lst]
        self.variance = variance(lst)
        self.mean = mean(lst)
        self.stdev = stdev(lst)

class Normal(Randomizer):
    def __init__(self, mean=0, stdev=1, iterations=50000):
        self.iterations = iterations
        self.mean_ = mean
        self.stdev_ = stdev
        super().__init__(Style.ITALIC + f'Нормальное распределение' + Style.END)

    def get_randoms(self):
        return normal(self.mean_, self.stdev_, self.iterations)


if __name__ == '__main__':
    border = '-' * 37
    for el in (TableMethod(), UniformFloat(0, 1), border,
               LCG(), UniformInt(0, 2239), border,
               Neumann(15, 50), border,
               CLT(0, 1), Normal()):
        print(el)
