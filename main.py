import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import integral_image
from tqdm import tqdm


def get_window_sum(cum_sum, w):
    window_sums = np.zeros((cum_sum.shape[0] - w[0] + 1, cum_sum.shape[1] - w[1] + 1))
    for i in tqdm(range(len(window_sums))):
        for j in range(window_sums.shape[1]):
            window_sums[i, j] = (
                cum_sum[i + w[0] - 1, j + w[1] - 1]
                - cum_sum[i + w[0] - 1, j]
                - cum_sum[i, j + w[1] - 1]
                + cum_sum[i, j]
            )
    return window_sums


def get_mean_std(image, w):

    """
    Западдим изображение, чтоб пройти по всем пикселям окном
    """

    pad_width = w // 2
    padded = np.pad(image, pad_width, mode="reflect")

    """
    Посчитаем куммулятивную сумму для стандартного изображения и изображения,
    в котором возведены в квадрат все значения
    """
    cum_sum = integral_image(padded, dtype=np.float64)
    padded = np.power(padded, 2)
    cum_sum_squared = integral_image(padded, dtype=np.float64)

    """
    Теперь получим среднее для каждого пикселя, для этого мы можем сделать следующее:
    у нас есть куммулятивные суммы, если взять правое нижнее значение в окне,
    вычесть из него правое верхнее и левое нижнее, то мы почти получим сумму в окне. 
    Проблема в том, что мы два раза вычли кумулятивную сумму для верхнего левого пикселя,
    поэтому добавим его, так мы получим корректную сумму для окна.
    Дальше остаётся только поделить на количество пикселей в окне, чтоб получить среднее.

    Точно также поступим для квадратов.
    """
    total_window_size = w ** 2
    w = (w, w)
    mean = get_window_sum(cum_sum, w)
    mean /= total_window_size
    squared_mean = get_window_sum(cum_sum_squared, w)
    squared_mean /= total_window_size

    """
    Теперь, когда у нас есть средние и средние квадратов, посчитаем стандартное 
    отклонение по формуле 
    Variance = E(x ** 2) - E(x) ** 2; 
    std = Variance ** (1 / 2)

    В данном случае клип нужен, тк из-за использования флотов при подсчётах не гарантируется,
    что squared_mean >= mean ** 2
    """
    std = np.sqrt(np.clip(squared_mean - mean ** 2, 0, None))
    return mean, std


def sauvola(image, window_size=15, k=0.2, r=None):
    """
    Порог вычисляется по формуле:
    T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))
    """
    if r is None:
        imin, imax = image.min(), image.max()
        r = 0.5 * (imax - imin)
    mean, std = get_mean_std(image, window_size)
    return mean * (1 + k * ((std / r) - 1))


if __name__ == "__main__":
    image = plt.imread("example.jpg")
    image = rgb2gray(image)
    mask = image > sauvola(image)
    image[mask] = 1
    image[~mask] = 0
    plt.imshow(image, cmap="Greys")
    plt.imsave('binarized.jpg', image)
    plt.waitforbuttonpress()
