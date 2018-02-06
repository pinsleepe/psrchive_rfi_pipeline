#!/usr/bin/env python3
"""
Visualise processed psrchive observations
@author: M Obrocka
./rfi_reduction.py --dest '/home/monika/Transcend/data/meerKAT/rfi_test' --plot True
"""
import argparse
from os import listdir
from os.path import join, isfile
import h5py
from astropy.time import Time
import pandas as pd
import calmap
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def all_files(datadir):
    return [f for f in listdir(datadir)
            if isfile(join(datadir, f))]


def h5_files(allfiles):
    return [f for f in allfiles
            if f.split('.')[-1] == 'h5']


def exclude_noise_diod(file_list):
    return [f for f in file_list
            if f.split('.')[0].split('_')[-1] != 'R']


def percentage(part, whole):
    return 100 * float(part) / float(whole)


class H5File(object):
    def __init__(self, hfile_path):
        self.path = hfile_path
        self.file = h5py.File(hfile_path, 'r')
        self.target = self.file['psrchive'].attrs['Target'].decode("utf-8")
        self._unify_target_names()
        self.mjd = self.file['psrchive']['MJD'][0]
        self.utc = Time(self.mjd, format='mjd', scale='utc')
        self.time = self.utc.datetime.strftime('%Y-%m-%d %H:%M')
        self.rfi = self.file['psrchive']['Mask'].attrs['RfiPercentage']
        self.hour_angle = self.file['psrchive']['HourAngle'][0]
        self.file_name = self.file['psrchive'].attrs['SourceName']
        self.num_subint = self.file['psrchive'].attrs['NumSubint']

    def _unify_target_names(self):
        target = self.target.replace('+', '-')
        target = target.replace(' ', '-')
        self.target = target

    def combine_info(self):
        info = [self.time, self.rfi, self.file_name,
                self.hour_angle, self.target, self.num_subint]
        return info

    def summarise_rfi_per_channel(self, channel_cutoff):
        mask = self.file['psrchive']['Mask'][:]
        rows = (mask == 0).sum(1)
        rfi_perc = [percentage(r, mask.shape[1]) for r in rows]
        rfi_vector = [0 if r > channel_cutoff else 1 for r in rfi_perc]
        return rfi_vector


class DFRfi(object):
    def __init__(self, info_array):
        r = [float(k) for k in info_array[:, 1]]
        ha = [float(k) for k in info_array[:, 3]]
        si = [int(k) for k in info_array[:, 5]]
        d = {'Date': pd.to_datetime(info_array[:, 0]),
             'RFI': pd.Series(r),
             'File': pd.Series(info_array[:, 2]),
             'HourAngle': pd.Series(ha),
             'Target': pd.Series(info_array[:, 4]),
             'SubInt': pd.Series(si)}
        self.df = pd.DataFrame(d)

    def write_csv(self, write_path):
        self.df.to_csv(write_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Visualise RFI info from reduced psrchives.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--plot',
        type=bool,
        dest='plot',
        action='store',
        default=False,
        help='Plot results?')
    parser.add_argument(
        '--dest',
        dest='dest',
        type=str,
        action='store',
        default='',
        help='path to directory with .h5 files')
    args = parser.parse_args()

    try:
        if args.dest:
            dest = args.dest
        else:
            print('Please tell me where the files are!!!')
    except NameError:
        raise

    allfiles = all_files(dest)
    h5files_full = h5_files(allfiles)
    h5files = exclude_noise_diod(h5files_full)
    print('Found %d .h5 files with RFI info.' % len(h5files))

    row = []
    for f in h5files:
        h = H5File(join(dest, f))
        row.append(h.combine_info())

    # create dataframe
    infoarray = np.asarray(row)
    df = DFRfi(infoarray)

    print('Unique tragets: %s' % df.df.Target.unique())
    print('In total: %d' % df.df.Target.unique().shape[0])

    # Drop observation with less than 30 subints to
    # get rid of most of the outliers.
    df.df = df.df.drop(df.df[df.df.SubInt < 30].index)

    if args.plot:
        print('Plot subints')
        df.df['SubInt'].plot.hist(alpha=0.5, bins=20)
        plt.title('Histogram of Subintegration Spread', fontsize=15)
        plt.show()

        print('Plot RFI percentage')
        df.df['RFI'].plot.hist(alpha=0.5, bins=50)
        plt.title('Histogram of RFI', fontsize=15)
        plt.show()

        # Give quick statistics.
        print(df.df.describe())

        # Count the number of observations per day and plot it.
        ht = pd.DataFrame({'count': df.df.groupby(df.df.Date.dt.date).size()}).reset_index()
        ht.index = pd.to_datetime(ht.Date)
        events = pd.Series(ht['count'].values, index=ht.index)

        plt.figure(figsize=(20, 20))
        calmap.yearplot(events, year=2017)
        plt.title('Total Observations Daily. Min=%d Max=%d' % (ht['count'].values.min(),
                                                               ht['count'].values.max()),
                  fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.show()

        # Plot summary RFI mask for all observations sorted by date.
        ch_cutoff = 80
        num_channels = 4096
        rfi_array = np.empty((num_channels, len(h5files)))
        idx = 0
        print('Creating summary RFI mask. This make take a minute...')
        for f in h5files:
            h = H5File(join(dest, f))
            rfi_vec = np.array(h.summarise_rfi_per_channel(ch_cutoff))  #[..., None]  # None keeps (n, 1) shape
            rfi_array[:, idx] = rfi_vec
            idx += 1

        plt.figure(figsize=(15, 10))
        plt.imshow(rfi_array, aspect="auto",
                   cmap=plt.cm.gray)
        plt.xlabel('File index', fontsize=20)
        plt.ylabel('Channel number', fontsize=20)
        plt.title('Coast guard mask', fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.show()



