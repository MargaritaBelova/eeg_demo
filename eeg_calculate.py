import numpy as np
from scipy.stats import t
from in_out import DataFragment, ProcessingConstants


class EEGfragments:
    """make 64 eeg fragments and time fragment from DataFragment which contains data from 64 electrodes"""
    __num_of_electrodes = 64

    def __init__(self, datafragment: DataFragment):
        dataFragment = datafragment.get_fragment()
        self._time = dataFragment[:, 0]
        self._EEGfragments = dataFragment[:, 1: EEGfragments.__num_of_electrodes + 1]
        self._length = dataFragment.shape[0]

    def get_electrode(self, electrode: int):  # electrode number is from 0 to 63 (inclusive)
        return self._EEGfragments[:, electrode]

    def get_time(self):
        return self._time

    def get_length(self):
        return self._length


class Histogram:
    """it builds the histogram from time series and counts parameters for  further calculations"""
    __slots__ = ('datasize', 'pdf', 'cdf', 'bins', 'accuracy')
    def __init__(self, data: np.ndarray, num_of_bins):
        self.datasize = data.size
        assert data.size != 0, 'Histogram: data.size cannot be 0'
        self.pdf, self.bins = np.histogram(data, bins=num_of_bins, density=True)
        bin = self.bins[1] - self.bins[0]
        self.pdf = self.pdf * bin       # make a probabilty for a cell in the cell
        self.cdf = np.cumsum(self.pdf)
        
        def clc_sample_std_deviation(pdf):
            sample_std_dev = 0
            for x in pdf:
                sample_std_dev += (x * (1 - x)) ** 0.5
            return sample_std_dev

        # todo check for zero sample_std_dev
        sample_std_dev = clc_sample_std_deviation(self.pdf)
        self.accuracy = data.size ** 0.5 / sample_std_dev
        self.cdf = np.cumsum(self.pdf)
        self.cdf[-1] = 1.       # correction of bad last value


class Autohistogram(Histogram):
    """it calculates optimal number of bins automatically"""
    def __init__(self, data: np.ndarray):
        num_of_bins = self.find_number_of_bins(data)
        super().__init__(data, num_of_bins)

    @staticmethod
    def find_number_of_bins(data: np.ndarray):
        # n1, n2 = 1, data.size
        # found that optimal value will be always around 20, so set n1=10  initially
        n1, n2 = 10, data.size

        dif_cache = [0, 0]
        while n2 - n1 > 1:
            n = (n1 + n2) // 2
            eps = 1 / n
            accuracy = Histogram(data, num_of_bins=n).accuracy
            # Approximation could be used for testing because it maybe faster
            # phi = Approximate_T_statistic.get_quantil_approx_div_by_eps(eps)
            phi = T_statistic.get_quantil_div_by_eps(eps=eps, freedom_degree=data.size)
            dif = phi - accuracy
            if dif > 0:
                n2 = n
                dif_cache[1] = dif
            elif dif < 0:
                n1 = n
                dif_cache[0] = dif
            else:
                n2 = n
                return n2

        if abs(dif_cache[1]) > abs(dif_cache[0]):
            return n1
        return n2


class T_statistic:
    """calculate t-criteria for n degrees of freedom: t_n(1-eps)"""

    def __init__(self, freedom_degree, epsilons):
        self._freedom_degree = freedom_degree
        #todo: it will be not useful later!
        #self._epsilons = epsilons
        quantils = np.array(list((t.ppf(1 - eps/2, df=self._freedom_degree, loc=0, scale=1) for eps in epsilons)))
        self._quantils_div_by_eps = quantils / epsilons

#    def get_epsilons(self):
#        return self._epsilons

    @staticmethod
    def get_quantil_div_by_eps(eps, freedom_degree):
        return t.ppf(1 - eps / 2, df=freedom_degree, loc=0, scale=1) / eps

    def get_quantils_div_by_eps(self):
        return self._quantils_div_by_eps


# it's faster than T_statistic class as uses approximations. If the presicion is enough, it's be better to use this class
class Approximate_T_statistic:
    """calculate t-criteria: t(1-eps); uses fast approximation instead of t function from scipy.
    Approximation is good if degree of freedom is large, e.g. >300-500 """

    def __init__(self, epsilons):
        # todo maybe it will be not useful later
        self._epsilons = epsilons
        quantils_approximated = np.array(list((-np.pi / 2 * np.log(1 - (1 - eps) ** 2)) ** 0.5 for eps in self._epsilons))
        self._quantils_approx_div_by_eps = quantils_approximated / self._epsilons

    def get_epsilons(self):
        return self._epsilons

    def get_quantils_approx_div_by_eps(self):
        return self._quantils_approx_div_by_eps

    @staticmethod
    def get_quantil_approx_div_by_eps(eps):
        return (- np.pi / 2 * np.log(1 - (1 - eps) ** 2)) ** 0.5 / eps



def calc_dif(autohistogram1: Autohistogram, autohistogram2: Autohistogram) -> (np.ndarray, np.ndarray):
    """C1 norm for two CDF histogram; CDF because right edges in hist1, hist2 are manually defined here as 1, left as 0"""
    bins1, bins2 = autohistogram1.bins, autohistogram2.bins
    merged_bins = np.union1d(autohistogram1.bins, autohistogram2.bins)
    merged_bins.sort()
    len_merged = merged_bins.size
    difference = np.empty(len_merged - 1)
    cdf1, cdf2 = autohistogram1.cdf, autohistogram2.cdf
    assert (cdf1.size == bins1.size - 1) and (cdf2.size == bins2.size - 1), \
        '{}: autohistogram size != bins size - 1'.format(calc_dif.__name__)

    if bins1[-1] < bins2[0]:
        difference = np.concatenate((abs(cdf1), np.array([1]), abs(1 - cdf2)), axis=0)
        return merged_bins, difference
    elif bins2[-1] < bins1[0]:
        difference = np.concatenate((abs(cdf2), np.array([1]), abs(1 - cdf1)), axis=0)
        return merged_bins, difference

    j1, j2 = 0, 0
    if bins1[0] == bins2[0]:
        difference[0] = abs(cdf1[0] - cdf2[0])
    else:
        while bins1[0] > merged_bins[j2]:
            difference[j2] = abs(cdf2[j2])
            j2 += 1
        else:
            while bins2[0] > merged_bins[j1]:
                difference[j1] = abs(cdf1[j1])
                j1 += 1
    left_edge = max(1, j1, j2)
    if j2 > 0:
        j2 -= 1
    elif j1 > 0:
        j1 -= 1
    for k in range(left_edge, len_merged - 1):
        x = merged_bins[k]
        if x >= bins1[j1+1]:
            j1 += 1
        if x >= bins2[j2+1]:
            j2 += 1
        if j1 == bins1.size - 1:
            for k1 in range(k, len_merged - 1):
                difference[k1] = abs(y1 - cdf2[j2 + k1 - k])
            break
        if j2 == bins2.size - 1:
            for k2 in range(k, len_merged - 1):
                l = j2 + k2 - k
                difference[k2] = abs(y2 - cdf1[j1 + k2 - k])
            break
        y1, y2 = cdf1[j1], cdf2[j2]
        difference[k] = abs(y1 - y2)
    return merged_bins, difference


class C1norm_sequence:
    def __init__(self, number_samples, constants: ProcessingConstants):
        self.__sample_num = number_samples
        self.constants = constants
        self.__sequence = self.__calc_statistical_gap()

    def get_C1norm_sequence(self):
        return self.__sequence

    def __calc_statistical_gap(self):
        dataFragment = DataFragment(self.constants.statistic_gap)
        sequence = np.empty(self.__sample_num)
        i = 0
        while i < self.__sample_num:
            dataFragment.read_file()
            eeg_fragment = EEGfragments(dataFragment)
            stat_gap = Statistic_gap(eeg_fragment.get_electrode(electrode=self.constants.electrode), self.constants.sliding_gap, self.constants.sliding_step)
            sequence[i] = self.__find_stationary_dist(stat_gap.gap_cdf_bins, stat_gap.gap_cdf)
            i += 1
            print('{} stationary distance is calculated'.format(i))

        print()
        return sequence

    def __find_stationary_dist(self, bins: np.ndarray, cdf: np.ndarray):
        difference = abs(cdf + bins[0: bins.size - 1] - 1)
        ind = np.argmin(difference)
        stationary_point = (bins[ind] + bins[ind + 1]) / 2  # uniform distribution in the bin
        return stationary_point


class Statistic_gap:
    """operates on one sample"""
    def __init__(self, statistic_gap: np.ndarray, sliding_gap_length: int, sliding_step: int):
        self._gap = statistic_gap
        self._sliding_len = sliding_gap_length
        self._sliding_step = sliding_step
        gap_hist = Autohistogram(self.__find_sliding_gaps_c1norm_seq())
        self.gap_cdf_bins, self.gap_cdf = gap_hist.bins, gap_hist.cdf

    def __find_sliding_gaps_c1norm_seq(self):
        num = (self._gap.size - 2 * self._sliding_len) // self._sliding_step
        c1norms = np.empty(num)
        i = 0
        while i < num:
            start = i * self._sliding_step
            finish1 = start + self._sliding_len
            finish2 = start + 2 * self._sliding_len
            sliding_gap1 = Autohistogram(self._gap[start: finish1])
            sliding_gap2 = Autohistogram(self._gap[finish1: finish2])
            bins, difference = calc_dif(sliding_gap1, sliding_gap2)
            c1norms[i] = difference.max()
            i += 1
        return c1norms


def test_L1norm_stationarity(sequence):
    """for stationary process result should be < 0.4"""
    middle = sequence.size // 2
    hist1, hist2 = Autohistogram(sequence[:middle]), Autohistogram(sequence[middle:])
    bins, difference = calc_dif(hist1, hist2)
    return (np.diff(bins) * difference).sum()
