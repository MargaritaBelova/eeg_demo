import draw_graphs
import eeg_calculate
from in_out import DataFragment, ProcessingConstants, Filenames, InputFileLen
import numpy as np

electrode = 0   # the number of processed electrode; the input file contains data from 64 electrodes;
shift = 100  # for precise results shift should be about 5, but the calculations would be longer
np.set_printoptions(precision=None, threshold=np.inf, suppress=True, linewidth=np.inf)

def EEG_stationarity_check():
    """ checks the original EEG series for stationarity before calculations: draws graph to show
    that the EEG series and its autocorrelation function is typical for non-stationary time series"""
    datafrag = DataFragment(2000)
    datafrag.read_file()
    sequence = datafrag.get_fragment()[:, 1]
    draw_graphs.draw_EEG_sample(seq=sequence, electrode_num=0)
    seq_title = "electrode {} data".format(electrode)
    draw_graphs.test_autocorr_stationarity_shift(sequence, max_lag=400, seq_title=seq_title)

EEG_stationarity_check()


# find optimal sample length for further calculations
constants = ProcessingConstants(sliding_step=shift, electrode=electrode)
#calculates how many pieces do we divide eeg into
number_samples = InputFileLen.get_sample_number(sample_len=constants.statistic_gap)

# main calculations: divides EEG into samples, moves sliding window along every sample with step = shift
# and finds the stationary distance of every sample using Kolmogorov-Smirnov test
c1norm = eeg_calculate.C1norm_sequence(number_samples, constants)
sequence = c1norm.get_C1norm_sequence()
sequence = sequence.reshape(1, sequence.size)

# write calculated sequence of stationary points to file
with open(Filenames.filenames['stationary_point_seq'], 'w') as file:
    np.savetxt(file, sequence, fmt='%12.6f')
    print("sequence of stationary points is written to {}".format("Filenames.filenames['stationary_point_seq']"))


sequence = np.loadtxt(Filenames.filenames['stationary_point_seq'])

# draw calculated stationary sequence - it's main result of EEG processing
draw_graphs.draw_stationary_point_seq(filename=Filenames.filenames['stationary_point_seq'],\
            electrode=electrode, shift=shift)
# as we haven't enough data to perform autocorrelation test for stationarity, we perform L1 test
# to demonstarate that calculated sequence is stationary; for stationary process result should be < 0.4
print("stationarity check using L1 norm distance:", eeg_calculate.test_L1norm_stationarity(sequence))
# and also perform test using dispersion: for stationary series it doesn't vary a lot with time
draw_graphs.test_variance_stationarity(sequence=sequence, seq_title='stationary point sequence of EEG series')


