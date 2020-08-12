import numpy as np
from matplotlib import pyplot as plt

def draw_histogram(bins: np.ndarray, histogram: np.ndarray, title=None, xlabel=None, ylabel=None, style='g:'):
    plt.figure(num=None, figsize=(16, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(bins[:-1], histogram, style)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.grid()
    plt.show()  


def draw_stationary_point_seq(filename, electrode, shift, level=None):
    with open(filename) as file:
        seq = np.loadtxt(file)
        plt.figure(num=None, figsize=(20, 5), dpi=90, facecolor='w', edgecolor='k')
        plt.plot(np.arange(seq.size), seq, 'g:', linewidth=2)
        if level:
            plt.plot([0, seq.size - 1], [level, level], 'r-', linewidth=1)
        plt.title('electrode №{}: Stationary point sequence, {} points; shift={} '.format(electrode, seq.size, shift), fontsize=14)
        plt.grid()
        plt.ylabel('stationary point', fontsize=14)
        plt.xlabel("gap's number", fontsize=14)
        plt.show()

# todo: delete it
"""
def draw_stationary_point_seq_pdf(filename, electrode, shift=5, bins=9):
    # draw pdf for stationary points sequence
    with open(filename, 'r') as file:
        seq = np.loadtxt(file)  # 78 points
    plt.hist(seq, bins=bins, density=True)
    plt.xlabel('$\\rho*$')
    plt.ylabel('$f(\\rho)$')
    plt.title('electrode №{}: pdf for {} stationary points, shift={}; {} bins'.format(electrode, seq.size, shift, bins))
    plt.grid()
    plt.show()
"""

def draw_EEG_sample(seq, electrode_num, save_fig=False):
    figure = plt.figure(num=None, figsize=(20, 6))
    plt.plot(range(seq.size), seq)
    plt.xlabel('time step', fontsize=16)
    plt.ylabel('voltage', fontsize=16)
    plt.title('EEG series, electrode №{}'.format(electrode_num), fontsize=16)
    plt.grid()
    plt.show()


def test_variance_stationarity(sequence, seq_title=None):
    variance = np.empty(sequence.size)
    for i in range(1, sequence.size):
        variance[i] = sequence[:i].var()
    fig = plt.figure(num=None, figsize=(12, 5))
    plt.plot(range(sequence.size), variance)
    plt.xlabel('$t$ (segment length)', fontsize=16)
    plt.ylabel('$\sigma^2(t)$', fontsize=16)
    if seq_title:
        plt.title('Stationarity check for {}:$\quad\sigma^2(t)$ for $\,X(t_0 + t),\, t \in (1, {})$'.format(seq_title, sequence.size), fontsize=15)
    else:
        plt.title('Stationarity check:$\quad\sigma^2(t)$ for $\,X(t_0 + t),\, t \in (1, {})$'.format(sequence.size), fontsize=16)
    plt.grid()
    plt.show()


def test_autocorr_stationarity(sequence, max_lag, seq_title=None, sample_size=None):
    if not sample_size:
        sample_size = sequence.size // 2
    seq1 = sequence[:sample_size]
    autocor = np.empty(max_lag)
    for lag in range(max_lag):
        seq2 = sequence[lag: sample_size + lag]
        autocor[lag] = ((seq1 - seq1.mean()) * (seq2 - seq2.mean())).mean() / (seq1.var() * seq2.var()) ** 0.5
    plt.figure(num=None, figsize=(15, 4), dpi=90, facecolor='w', edgecolor='k')
    plt.plot(range(1, max_lag), autocor[1:])
    plt.title('Stationarity check: autocorrelation function for {} (excluding lag=0)'.format(seq_title), fontsize=15)
    plt.xlabel('lag', fontsize=14)
    plt.ylabel('correlation', fontsize=14)
    plt.grid()
    plt.show()


def test_autocorr_stationarity_shift(sequence, max_lag, sample_size=None, seq_title=None):
    if not sample_size:
        sample_size = sequence.size // 2
    autocor = np.empty(max_lag)
    plt.figure(num=None, figsize=(15, 5), dpi=90, facecolor='w', edgecolor='k')
    for start in range(0, 1000, 400):
        seq1 = sequence[start: sample_size + start]
        for lag in range(max_lag):
            seq2 = sequence[lag: sample_size + lag]
            autocor[lag] = ((seq1 * seq2).mean() - seq1.mean() * seq2.mean()) / (seq1.var() * seq2.var()) ** 0.5
        plt.plot(range(0, max_lag), autocor[:], label='start={}'.format(start))

    plt.title('Stationarity check: autocorrelation function for {}'.format(seq_title), fontsize=15)
    plt.xlabel('lag', fontsize=14)
    plt.ylabel('correlation', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


