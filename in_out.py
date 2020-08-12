import numpy as np
from os import path


class Singleton:    # serves as parent for singleton in/out classes
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance


class ProcessingConstants(Singleton):
    """after experimental calculations these parameters were found as the optimal ones;
        the class gets one external parameter -- sliding step)"""
    def __init__(self, sliding_step: int, electrode: int):
        super().__setattr__('statistic_gap', 30_000)
        super().__setattr__('sliding_gap', 5000)
        super().__setattr__('sliding_step', sliding_step)
        if (self.statistic_gap - 2 * self.sliding_gap) % self.sliding_step:
            raise ValueError('bad sliding_step value: statistic_gap - sliding_gap) % sliding_step have to be 0')
        super().__setattr__('electrode', electrode)

    def __setattr__(self, key, value):
        print('{} is not allowed to change'.format(key))


class InputFileLen(Singleton):
    """as we work with one big file, number of strings was determined once before processing"""
    # input file 'input_files/data_first_part.txt' contains 2340001 lines including header
    #__file_length = 2_340_000
    __file_length = 400_000

    def __init__(self, *file_len):
        self.__file_length = file_len[0] or self.__file_length

    @classmethod
    def get_file_length(cls):
        return cls.__file_length

    @classmethod
    def get_sample_number(cls, sample_len):
        """returns how many pieces do we divide eeg into"""
        residue = cls.__file_length % sample_len
        if residue:
            print('WARNING: bad sample_len: cls.__file_length % sample_len should is not equal to 0; {} closing points will not be processed'.format(residue))
        return cls.__file_length // sample_len


class Filenames(Singleton):
    filenames = {
        'eeg': path.join('input_files', 'eeg_short_sample.txt'),
        'stationary_point_seq': path.join('output_files', 'stationary_points_sequence_electrode0.txt')
    }


class DataFragment(Singleton):
    """read a piece of particular size of EEG txt file; file contains 64 electrodes + time column"""
    __eeg_filename = Filenames.filenames['eeg']
    __input_file_pos = 0
    __number_of_columns = 65

    def __new__(cls, *args):
        if cls.instance is None:
            with open(cls.__eeg_filename) as input_file:
                ignored_first_line = input_file.readline()
                cls.__input_file_pos = input_file.tell()
                super().__new__(cls)
        return cls.instance

    def __init__(self, fragment_length, input_filename=__eeg_filename):
        self._fragment_length = fragment_length
        self.__fragment = np.empty((self._fragment_length, self.__number_of_columns))
        self.__input_filename = input_filename

    def __str__(self):
        return str(self.__fragment)

    @classmethod
    def get_number_of_columns(cls):
        return cls.__number_of_columns

    def get_fragment(self):
        return self.__fragment

    def read_file(self):
        with open(self.__input_filename, 'r') as input_file:
            input_file.seek(self.__input_file_pos)
            i = 0
            while i < self._fragment_length:
                line = input_file.readline()
                data_line = tuple(float(x) for x in line.split(','))
                self.__fragment[i] = data_line
                i += 1
            DataFragment.__input_file_pos = input_file.tell()

    def get_file_pos(self):
        return self.__input_file_pos
