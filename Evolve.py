import numpy as np
import random
from skimage import io
from skimage.transform import rescale, resize
from skimage import img_as_ubyte
import time


class pop:

    def __init__(self, img, target):
        self.target = target
        self.img = img

    def cal_val(self):
        self.val = np.sum(np.abs(np.subtract(self.target, self.img)))

    @staticmethod
    def generate_pop(size, target):  # return a list of randomly generated population
        for i in range(size):
            a = pop(np.random.randint(0, 256, target.shape), target)
            a.cal_val()
            yield a

    def mutaion(self):
        self.img[random.randint(0, self.img.size - 1)] = random.randint(0, 256)

    def crossover(self, p2):
        pos = random.randint(1, self.img.size)
        return pop(np.concatenate((self.img[:pos], p2.img[pos:])), self.target)


def saveimg(v, img):
    io.imsave(v, img)


def showimg(img):
    io.imshow(img)


def evolve(target):
    Population = pop.generate_pop(100, target)
    Population = sorted(Population, key=lambda e: e.val)
    new_pop = [None] * 100
    set1 = Population[:10]
    set2 = Population[:10]
    limit = 1000
    index = 0
    best_child = Population[0]
    while (index < limit and best_child.val > 1000):
        for i in range(100):
            p1 = set1[int(random.random() * 10)]
            p2 = set2[int(random.random() * 10)]
            child = p1.crossover(p2)
            if int(random.random() * 10) == 1:
                child.mutaion()
            child.cal_val()
            new_pop[i] = child
        Population = new_pop
        Population = sorted(Population, key=lambda e: e.val)
        if Population[0].val < best_child.val:
            # print("New best child found at generation {} with cost {}".format(index, child.val))
            best_child = Population[0]
        index += 1
        set1 = Population[:10]
        set2 = Population[:10]

    return best_child


def main():
    messi = io.imread('messi.png', as_gray=True)
    resized_messi = resize(messi, (75, 75), anti_aliasing=True)
    resized_messi = img_as_ubyte(resized_messi)
    og_shape = resized_messi.shape
    resized_messi = np.reshape(resized_messi, (resized_messi.size, 1))
    seconds1 = time.time()
    sample = evolve(resized_messi)
    seconds2 = time.time()
    print(seconds2 - seconds1)
    resized_messi = np.reshape(sample.img, og_shape)
    saveimg('messi_2.png', resized_messi)


main()
