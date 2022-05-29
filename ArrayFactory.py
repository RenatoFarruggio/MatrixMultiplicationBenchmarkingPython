from random import Random


class ArrayFactory:
    def __init__(self, size):
        self.size = size
        self.rand = Random()

    def get_new_dense_array(self):
        array = []
        for _ in range(self.size):
            line = []
            for __ in range(self.size):
                line.append(self.rand.random() * 10)
            array.append(line)
        return array

    def get_new_sparse_array(self):
        array = []
        for _ in range(self.size):
            line = []
            for __ in range(self.size):
                if self.rand.randint(0, 100) <= 90:
                    line.append(0)
                else:
                    line.append(self.rand.random() * 10)
            array.append(line)
        return array


if __name__ == '__main__':
    af = ArrayFactory(2)
    print(af.get_new_dense_array())
