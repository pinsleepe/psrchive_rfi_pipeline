import matplotlib as mpl
# for Docker use
mpl.use('Agg')

import matplotlib.pyplot as plt
import logging
import logging.handlers
import pysolr
import collections
import sys
from os.path import join, isfile, exists, getsize
from os import makedirs
import requests
import psrchive as psr
from coast_guard import cleaners
import time
import h5py
import numpy as np
from shutil import rmtree
import csv
from itertools import izip
from astroplan import Observer
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS
from astropy.coordinates import Angle
from astroplan import download_IERS_A
download_IERS_A()

LOGGER = logging.getLogger(__name__)

# some constants
ONE_KB = 1024
ONE_MB = ONE_KB * ONE_KB

static_mask = ['0:210',
               '286:497',
               '831:886',
               '936:938',
               '1099:1101',
               '1108:1111',
               '1113:1128',
               '1166:1167',
               '1175:1177',
               '1180:1181',
               '1184:1186',
               '1194:1196',
               '1199:1201',
               '1204:1205',
               '1223:1224',
               '1228:1229',
               '1276:1277',
               '1295:1296',
               '1300:1301',
               '1309:1311',
               '1333:1335',
               '1338:1339',
               '1347:1349',
               '1357:1358',
               '1362:1363',
               '1387:2128',
               '2700:2709',
               '2900:3044',
               '3174:3687',
               '3896:4095']


def _create_static_mask(idx):
    obs_mask = [static_mask[e] for e in idx]
    return ';'.join(obs_mask)


# TODO badchannels as an input
def coast_guard_config_string(cleaner_type, static=None):
    """
    Build config string for coast guard
    """
    # Create a quick field+query type
    query_list = []
    fq = collections.namedtuple('field_query', 'field query')
    if cleaner_type == "surgical":
        query_list = [
            #
            fq('chan_numpieces', 1),
            #
            fq('subint_numpieces', 1),
            #
            fq('chanthresh', 3),
            #
            fq('subintthresh', 3)]

    elif cleaner_type == 'rcvrstd':
        if static:
            badchans = _create_static_mask(static)
        else:
            badchans = _create_static_mask([0, -1])
        query_list = [
            #
            fq('badfreqs', None),
            #
            fq('badsubints', None),
            #
            fq('trimbw', 0),
            #
            fq('trimfrac', 0),
            #
            fq('trimnum', 0),
            #
            fq('response', None),
            #
            fq('badchans', badchans)]

    # Construct the query
    return ','.join('%s=%s' % (fq.field, fq.query) for fq in query_list)


class ObsQuery(object):
    def __init__(self, product='PTUSETimingArchiveProduct'):
        """

        :param product: PulsarTimingArchiveProduct, PTUSETimingArchiveProduct
        :param verbose: display print messages
        :param logger:
        """
        LOGGER.info('Created ObsQuery object')
        self.product = product
        self.search = None
        self.solr_url = 'http://192.168.1.50:8983/solr/kat_core'

    def standard_observation_query(self, days):
        """
        Build query message
        :param days: int
        :return: str
        """
        LOGGER.info('Querying product: %s.' % self.product)
        # Create a quick field+query type
        fq = collections.namedtuple('field_query', 'field query')

        query_list = [
            # Only want MeerKAT AR1 Telescope Products
            fq('CAS.ProductTypeName', self.product),
            # Observations from the last 3 days
            fq('StartTime', '[NOW-%dDAYS TO NOW]' % days)]

        # Construct the query
        return ' AND '.join('%s:%s' % (fq.field, fq.query)
                            for fq in query_list)

    def paged_search(self, days, query=None, rows=10):
        """
        Deep paging for a large start offset into the search results.
        The cursorMark parameter allows efficient iteration
        over a large result set.
        :param days: integer
        :param query: str
        :param rows: integer
        :return: yields only 'rows' results at a time (generator)
        """
        LOGGER.info('Paged query. Querying product: %s.' % self.product)
        solr = pysolr.Solr(self.solr_url)
        self.search = query if query is not None else self.standard_observation_query(days)
        LOGGER.info('Paged query. Search query: %s.' % self.search)
        currentCursorMark = '*'
        res = solr.search(self.search, rows=rows, sort='id asc',
                          **{'cursorMark': '%s' % currentCursorMark})
        LOGGER.info('Paged query. Found %d results.' % res.hits)
        while currentCursorMark != res.nextCursorMark:
            currentCursorMark = res.nextCursorMark
            res = solr.search(self.search, rows=rows, sort='id asc',
                              **{'cursorMark': '%s' % res.nextCursorMark})
            yield res

    def query_recent_observations(self, days=3, query=None):
        """
        Basic paging
        :param days:
        :param query:
        :return:
        """

        self.search = query if query is not None else self.standard_observation_query(days)
        LOGGER.info('Standard query. Querying product: %s.' % self.product)
        LOGGER.info('Standard query. Search query: %s.' % self.search)
        archive = pysolr.Solr(self.solr_url)
        results = archive.search(self.search, sort='StartTime desc', rows=1000)
        LOGGER.info('Standard query. Found %d results.' % results.hits)
        return results


class PulsarArchive(object):
    def __init__(self, observation, index):
        self.index = index
        LOGGER.info('PulsarArchive %i created' % index)
        self.observation = observation
        try:
            self.block_id = observation['ScheduleBlockIdCode']
        except:
            self.block_id = 0
            LOGGER.error('No Schedule Block Id Code available for PulsarArchive %i!' % index)

        self.archive_name = observation['Filename']
        LOGGER.info('Analysing archive %s.' % self.archive_name)
        self.remote_location = None
        self.arch_files_remote = None
        self.arch_files_local = []
        self.arch_urls = None
        self.ar_files = None
        self.psr_archive = None
        self.lowFreq = None
        self.highFreq = None
        self.sourceName = None
        self.nChan = None
        self.rfi_occupancy = None
        self.local_archive_path = None
        self.period = None
        self.mjd_subint = None
        self.ha_subint = None
        self.azalt_subint = None
        self.target = None
        self.observer = observation['Observer']
        self.header = False
        self.ra = None
        self.dec = None
        self.skip = False
        self.static_mask = None
        self.static_rfi_occupancy = None

        lat = -30.7110555556117
        lon = 21.4438888892753
        el = 1035
        meerkat = Observer(longitude=lon * u.deg,
                           latitude=lat * u.deg,
                           elevation=el * u.m,
                           name="MeerKAT")
        self.meerkat = meerkat

        self._init()

    def _init(self):
        LOGGER.info('Initialising PulsarArchive %i!' % self.index)
        self._get_archives_list()
        self._get_ar_list()
        self._get_archives_urls()

    def _get_archives_list(self):
        """
        Find psrchive files on the server
        :return:
        """
        archive_remote_files = self.observation['CAS.ReferenceDatastore'][1:-1]
        self.arch_files_remote = [f for f in archive_remote_files
                                  if f.split('.')[1] == 'ar']
        LOGGER.info('Found %d psrchive files in %s.' % (len(self.arch_files_remote),
                                                        self.archive_name))

    def _get_ar_list(self):
        """
        Remove data path for 'for loops' and sort
        :return:
        """
        self.ar_files = [l.split('/')[-1] for l in self.arch_files_remote]
        self.ar_files.sort()
        LOGGER.debug('Removing data path for "for loops" and sorting.')

    def _get_archives_urls(self):
        """
        Create psrchive files urls for download
        :return:
        """
        remote_location = self.observation['FileLocation'][0]
        remote_location = remote_location.replace('/var/kat',
                                                  'http://kat-archive.kat.ac.za', 1)
        self.remote_location = remote_location.replace('archive/',
                                                       'archive2/', 1)
        self.arch_urls = [join(remote_location, self.archive_name, a)
                          for a in self.ar_files]
        self.arch_urls.sort()
        LOGGER.debug('Creating urls.')

    def _writeDir(self, full_path):
        """
        Create directory for observation
        :param full_path:
        :return:
        """
        if not exists(full_path):
            makedirs(full_path)
            LOGGER.info('Creating directory for observation: %s' % full_path)
        # check if successful
        LOGGER.info('Directory for observation already exists! Path: %s' % full_path)
        return exists(full_path)

    def _get_header_file(self):
        """
        Download obs.header file from the server
        :return:
        """
        remote_files = self.observation['CAS.ReferenceDatastore'][1:-1]
        header_path = [f for f in remote_files if f.split('.')[1] == 'header']
        if header_path:
            self.header = True
            LOGGER.info('obs.header file exists for %s. Downloading.' % self.archive_name)
        else:
            LOGGER.error("obs.header file doesn't exists for %s. Skipping!" % self.archive_name)

    def download_file(self, filename, local_file_size, file_exists, url):
        """
        Snippet taken from VeerMeerKAT
        :param filename: str
        :param local_file_size: int
        :param file_exists: bool
        :param url: str
        :return:
        """
        LOGGER.info('Attempting to download %s.' % filename)
        # Infer the HTTP location from the KAT archive file location
        headers = {"Range": "bytes={}-".format(local_file_size)}
        if file_exists:
            r = requests.get(url, headers=headers, stream=True)
        else:
            r = requests.get(url, stream=True)
        # Server doesn't care about range requests and is just
        # sending the entire file
        if r.status_code == 200:
            LOGGER.info("Downloading '{}'".format(filename))
            remote_file_size = r.headers.get('Content-Length', None)
            file_exists = False
            local_file_size = 0
        elif r.status_code == 206:
            if local_file_size > 0:
                LOGGER.info("'{}' already exists, resuming download from {}.".format(filename,
                                                                                     local_file_size))

            # Create a fake range if none exists
            fake_range = "{}-{}/{}".format(local_file_size, sys.maxint,
                                           sys.maxint - local_file_size)

            remote_file_size = r.headers.get('Content-Range', fake_range)
            remote_file_size = remote_file_size.split('/')[-1]
        elif r.status_code == 416:
            LOGGER.info("'{}' already downloaded".format(filename))
            remote_file_size = local_file_size
        else:
            LOGGER.critical("HTTP Error Code {}".format(r.status_code))
            raise ValueError("HTTP Error Code {}".format(r.status_code))
        LOGGER.info('URL: %s File size: %s Status code: %s' % (url, remote_file_size, r.status_code))
        f = open(filename, 'ab' if file_exists else 'wb')
        # Download chunks of file and write to disk
        try:
            with f:
                downloaded = local_file_size
                for chunk in r.iter_content(chunk_size=ONE_MB):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        LOGGER.debug(downloaded)
        except KeyboardInterrupt as kbe:
            LOGGER.info("Quitting download on Keyboard Interrupt")
            pass

    def download_header(self):
        """
        run after download_observation
        """
        self._get_header_file()
        if self.header:
            header = 'obs.header'
            filename = join(self.local_archive_path, header)
            file_exists = exists(filename) and isfile(filename)
            local_file_size = getsize(filename) if file_exists else 0
            header_url = join(self.remote_location, self.archive_name, header)
            self.download_file(filename, local_file_size, file_exists, header_url)

    def download_observation(self, directory, limit):
        """
        Download psrchive files from the server
        :param directory: str
        :return:
        """
        LOGGER.info('Creating directory for %s.' % self.archive_name)
        # create obs folder
        local_archive_path = join(directory, self.archive_name)
        self.local_archive_path = local_archive_path
        LOGGER.info('Creating directory for %s with local path %s.' % (self.archive_name,
                                                                       local_archive_path))
        # check if directory exists, if not create one
        self._writeDir(local_archive_path)
        # loop through arch files
        if len(self.ar_files) == 0:
            LOGGER.warning('No psrchive files found in %s! No point continuing!'
                           'Skipping processing!' % self.archive_name)
            self.skip = True
            self.cleanup()
        elif len(self.ar_files) >= limit:
            LOGGER.warning('Limit reached! Observation %s has too many files!. '
                           'Skipping processing!' % self.archive_name)
            self.skip = True
            self.cleanup()
        else:
            for idx, url in enumerate(self.arch_urls):
                filename = join(local_archive_path, self.ar_files[idx])
                self.arch_files_local.append(filename)
                file_exists = exists(filename) and isfile(filename)
                local_file_size = getsize(filename) if file_exists else 0
                self.download_file(filename, local_file_size, file_exists, url)

    def _get_hour_angle(self, arch, ra, dec):
        """
        Calculate hour angle
        To get hms string:
        Angle(value * u.rad).hms
        :param arch: psrchive object
        :param ra: astropy.coordinates.Angle
        :param dec: astropy.coordinates.Angle
        :return: str ex. 3h51m34.4942s, MJD
        """
        LOGGER.debug('Calculate hour angle for archive %s.' % arch.get_filename().split('/')[-1])
        mjd = arch.get_Integration(0).get_start_time().in_days()
        t = Time(mjd, format='mjd', scale='utc')
        target = SkyCoord(ICRS, ra=ra, dec=dec, obstime=t)
        hr = self.meerkat.target_hour_angle(t, target).to_value(unit=u.rad)
        return hr

    def _get_azel(self, arch, ra, dec):
        LOGGER.debug('Calculate azimuth and altitude for archive %s.' % arch.get_filename().split('/')[-1])
        mjd = arch.get_Integration(0).get_start_time().in_days()
        t = Time(mjd, format='mjd', scale='utc')
        target = SkyCoord(ICRS, ra=ra, dec=dec, obstime=t)
        azel = self.meerkat.altaz(t, target=target)
        az = azel.az.to_value(unit=u.deg)
        alt = azel.alt.to_value(unit=u.deg)
        # return str(az)+';'+str(alt)
        return az, alt

    def gen_archive(self):
        """
        Process psrchive files
        """
        LOGGER.info('Gnerating archive files for observation %s.' % self.archive_name)
        self.arch_files_local.sort()
        # load psrchive objects
        archives = [psr.Archive_load(f.encode('ascii', 'ignore'))
                    for f in self.arch_files_local]
        # calculate ra and dec
        LOGGER.debug('Calculate ra and dec for observation %s.' % self.archive_name)
        if self.header:
            self.target = self.observation['Targets'][0]
            self.ra = archives[0].get_coordinates().ra().getHMS()
            self.dec = archives[0].get_coordinates().dec().getDMS()
        else:
            val = self.observation['KatpointTargets'][0].split(',')
            self.ra = val[-2]
            self.dec = val[-1]
            self.target = val[0]
        raA = Angle(self.ra, unit=u.hour)
        decA = Angle(self.dec, unit=u.deg)
        # calculate hour_angle and MJD per subint
        LOGGER.debug('Calculate hour_angle per subint '
                     'for observation %s.' % self.archive_name)
        self.ha_subint = [self._get_hour_angle(a, raA, decA)
                          for a in archives]
        LOGGER.debug('Calculate MJD per subint '
                     'for observation %s.' % self.archive_name)
        self.mjd_subint = [a.get_Integration(0).get_start_time().in_days()
                           for a in archives]
        # calculate observation start and stop time
        LOGGER.debug('Calculate start and stop time for '
                     'observation %s.' % self.archive_name)
        # calculate period
        LOGGER.debug('Calculate period for observation %s.' % self.archive_name)
        self.period = archives[0].get_Integration(0).get_folding_period()
        # append data to psrchive[0] for coast guard
        LOGGER.debug('Appending data to psrchive[0] for '
                     'observation %s.' % self.archive_name)
        for i in range(1, len(archives)):
            archives[0].append(archives[i])
        # archives[0].pscrunch()
        self.psr_archive = archives[0]
        LOGGER.debug('Calculate azimuth and altitude per subint '
                     'for observation %s.' % self.archive_name)
        nsint = self.psr_archive.get_nsubint()
        azel_array = np.empty((nsint, 2))
        idx = 0
        for a in archives:
            az, el = self._get_azel(a, raA, decA)
            azel_array[idx, :] = az, el
            idx += 1
        self.azalt_subint = azel_array
        # calculate top and bottom frequencies
        LOGGER.debug('Calculate top and bottom frequencies for '
                     'observation %s.' % self.archive_name)
        self.lowFreq = self.psr_archive.get_centre_frequency() - \
                       self.psr_archive.get_bandwidth() / 2.0
        self.highFreq = self.psr_archive.get_centre_frequency() + \
                        self.psr_archive.get_bandwidth() / 2.0
        LOGGER.debug('Read source name and number of channels '
                     'for observation %s.' % self.archive_name)
        self.sourceName = self.psr_archive.get_source()
        self.nChan = self.psr_archive.get_nchan()

    def run_coast_guard(self, cleaner_type='', config_string=None, static=None):
        """
        Run coast guard on psrchive file
        :param cleaner_type: surgical or rcvrstd
        :param config_string: str
        :return:
        """
        LOGGER.info('Running cleaner for observation %s.' % self.archive_name)
        cleaner = cleaners.load_cleaner(cleaner_type)
        config = config_string if config_string is not None \
            else coast_guard_config_string(cleaner_type, static=static)
        self.static_mask = config.split(',')[-1].strip('badchans=')
        cleaner.parse_config_string(config)
        LOGGER.info("Running coast quard in mode %s with string '%s'." % (cleaner_type,
                                                                          config))
        cleaner.run(self.psr_archive)
        return self.psr_archive.get_weights().T

    def count_rfi(self, rfi_array):
        """
        percentage of rfi channels
        """
        LOGGER.info('Calculating RFI occupancy for observation %s.' % self.archive_name)
        obs_mask = []
        for s in self.static_mask.split(';'):
            l = s.split(':')
            r = np.arange(int(l[0]), int(l[1]) + 1)
            nr = r.tolist()
            obs_mask.extend(nr)
        mask_idx = np.ones((self.nChan, self.psr_archive.get_nsubint()))
        mask_idx[obs_mask, :] = 0
        count_static = np.where(mask_idx == 0)[1].shape[0]
        count = np.where(rfi_array == 0)[1].shape[0]
        self.rfi_occupancy = (float(count - count_static)/float(rfi_array.size))*100
        self.static_rfi_occupancy = (float(count_static)/float(rfi_array.size))*100

    def process_header(self):
        """
        Write text file as a dictionary
        :return:
        """
        if self.header:
            LOGGER.info('Processing obs.header for observation %s.' % self.archive_name)
            header = 'obs.header'
            filename = join(self.local_archive_path, header)
            header_text = list(csv.reader(open(filename, 'rb'), delimiter='\t'))
            header_list = []
            for row in header_text:
                header_line = row[0].split(' ')
                small_list = filter(None, header_line)
                header_list.extend(small_list)
            i = iter(header_list)
            self.header = dict(izip(i, i))

    def plot_mask(self, array, save=False, directory=''):
        LOGGER.info('Plotting RFI mask for observation %s.' % self.archive_name)
        fig, ax1 = plt.subplots(1, 1,
                                figsize=[15, 10],
                                tight_layout="false")
        ax1.imshow(array,
                   origin="lower",
                   aspect="auto")
        ax1.set_title(self.archive_name)
        ax1.set_title("RFI mask", loc="left")
        ax1.set_ylabel("Channel number")
        ax1.yaxis.set_ticks(np.arange(0, self.nChan - 1, 200))
        ax1.set_xlabel("Subint number")
        ax1Secondary = ax1.twinx()
        ax1Secondary.set_ylabel("Frequency (MHz)")
        ax1Secondary.set_ylim(self.lowFreq, self.highFreq)
        ax1Secondary.yaxis.set_ticks(np.arange(self.lowFreq, self.highFreq, 25))
        if save:
            png_path = join(directory, self.archive_name + '_mask.png')
            fig.savefig(png_path)
            LOGGER.info('Writing png as %s for observation %s.' % (png_path,
                                                                   self.archive_name))
        return fig

    def plot_bandpass(self, array, save=False, directory=''):
        LOGGER.info('Plotting bandpass for observation %s.' % self.archive_name)
        fig, ax1 = plt.subplots(1, 1,
                                figsize=[15, 10],
                                tight_layout="false")
        plt.plot(array[0, :], array[1, :])
        ax1.set_title(self.archive_name)
        ax1.set_title("Bandpass", loc="left")
        ax1.set_ylabel("Power [arbitrary]")
        ax1.set_xlabel("Frequency [MHz]")
        if save:
            png_path = join(directory, self.archive_name + '_bandpass.png')
            fig.savefig(png_path)
            LOGGER.info('Writing png as %s for observation %s.' % (png_path,
                                                                   self.archive_name))
        return fig

    def write_h5(self, mask, bandpass0, bandpass1, directory=''):
        LOGGER.info('Writing results to H5 for observation %s.' % self.archive_name)
        h5_path = join(directory, self.archive_name + '.h5')
        LOGGER.info('Writing hdf5 file as %s' % h5_path)
        with h5py.File(h5_path, 'w') as hf:
            grp = hf.create_group("psrchive")
            dset0 = grp.create_dataset('Mask', data=mask, dtype='int8')
            bandpass = np.empty((2, 5, self.nChan))
            bandpass[0, :, :] = bandpass0
            bandpass[1, :, :] = bandpass1
            dset1 = grp.create_dataset('Bandpass', data=bandpass, dtype='f')
            dset1.attrs['D1'] = 'Freq, XX_mean, YY_mean, XX_var, YY_var'
            dset1.attrs['D0'] = '0: Bandpass before clean, 1: Bandpass after clean'

            grp.attrs['NumChan'] = self.nChan
            grp.attrs['NumSubint'] = self.psr_archive.get_nsubint()
            grp.attrs['SourceName'] = self.archive_name
            grp.attrs['RA'] = self.ra
            grp.attrs['Dec'] = self.dec
            grp.attrs['CentreFrequency'] = self.psr_archive.get_centre_frequency()
            grp.attrs['Bandwidth'] = self.psr_archive.get_bandwidth()
            grp.attrs['DM'] = self.psr_archive.get_dispersion_measure()
            grp.attrs['ObsDuration'] = self.psr_archive.integration_length()
            dset0.attrs['RfiPercentage'] = self.rfi_occupancy
            dset0.attrs['StaticRfiPercentage'] = self.static_rfi_occupancy
            dset0.attrs['StaticMask'] = self.static_mask

            grp.create_dataset('MJD', data=self.mjd_subint, dtype='f')
            grp.create_dataset('HourAngle', data=self.ha_subint, dtype='f')
            grp.create_dataset('AzimuthElevation', data=self.azalt_subint, dtype='f')

            grp.attrs['Period'] = self.period
            grp.attrs['Target'] = self.target.encode('ascii', 'ignore')
            grp.attrs['Observer'] = self.observer.encode('ascii', 'ignore')
            # header not available for normal timing archive
            if self.header:
                grp.attrs['AdcSyncTime'] = self.header['ADC_SYNC_TIME']
                grp.attrs['Antenna'] = self.header['ANTENNAE']
                grp.attrs['AntennaWeight'] = 1/np.sqrt(len(self.header['ANTENNAE'].split(',')))
                grp.attrs['Mode'] = self.header['MODE']

            grp.attrs['ScheduleBlockID'] = self.block_id.encode('ascii', 'ignore')

    def cleanup(self):
        """
        short fix, use multiprocessing
        """
        LOGGER.info('Cleaning up after observation %s!' % self.archive_name)
        self.arch_files_remote = None
        self.arch_files_local = []
        self.arch_urls = None
        self.ar_files = None
        self.psr_archive = None

        rmtree(self.local_archive_path)

    def estimate_bandpass(self):
        LOGGER.info('Calculating bandpass estimate for observation %s!' % self.archive_name)
        clone = self.psr_archive.clone()
        clone.tscrunch()
        subint = clone.get_Integration(0)
        (bl_mean, bl_var) = subint.baseline_stats()
        # choose only XX and YY
        bl_mean = bl_mean.squeeze()
        bl_mean = bl_mean[[0, 1], :]
        # choose only XX and YY
        bl_var = bl_var.squeeze()
        bl_var = bl_var[[0, 1], :]

        # choose only XX and YY
        # non_zeroes = np.where(bl_mean != 0.0)
        # non_zeroes = non_zeroes[1].reshape((2, self.nChan))
        # non_zeroes = non_zeroes[0, :]

        min_freq = clone.get_Profile(0, 0, 0).get_centre_frequency()
        max_freq = clone.get_Profile(0, 0, self.nChan - 1).get_centre_frequency()

        freqs = np.linspace(min_freq, max_freq, self.nChan)

        return np.vstack((freqs, bl_mean, bl_var))


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def print_time(self, time_elapsed):
        print("Time elapsed: %s" % time_elapsed)


if __name__ == "__main__":
    import logging
    import logging.handlers

    LOG_FILENAME = '/data/logging.out'
    f = logging.Formatter(fmt='%(levelname)s:%(name)s: %(message)s '
                              '(%(asctime)s; %(filename)s:%(lineno)d)',
                          datefmt="%Y-%m-%d %H:%M:%S")
    handlers = logging.handlers.RotatingFileHandler(LOG_FILENAME,
                                                    encoding='utf8',
                                                    maxBytes=100000,
                                                    backupCount=1),
    logging.StreamHandler()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for h in handlers:
        h.setFormatter(f)
        h.setLevel(logging.DEBUG)
        root_logger.addHandler(h)

    my_timer = Timer()

    obs_query = ObsQuery()
    logging.info('Running RFI mask!! \nStarted. \nUTC time:  %s' % my_timer.start)
    days = 5
    search = obs_query.query_recent_observations(days)
    local_directory = '/data'

    archive = PulsarArchive(search.docs[0], 0)
    archive.download_observation(local_directory, 100)
    if archive.skip:
        logging.info('File too big! Skipping!')
    else:
        archive.download_header()
        archive.process_header()
        archive.gen_archive()
        before_bp = archive.estimate_bandpass()
        weights = archive.run_coast_guard(cleaner_type="surgical")
        mask = archive.run_coast_guard(cleaner_type="rcvrstd")
        after_bp = archive.estimate_bandpass()
        archive.count_rfi(mask)
        bp = archive.estimate_bandpass()
        # fig = archive.plot_mask(mask,
        #                         save=True,
        #                         directory='/data')
        # fig.clear()
        # fig = archive.plot_bandpass(before_bp, save=True, directory='/data')
        # fig.clear()
        archive.write_h5(mask, before_bp, after_bp, directory='/data')
        archive.cleanup()
        time_hhmmss = my_timer.get_time_hhmmss()
        my_timer.print_time(time_hhmmss)
        logging.info('Elapsed time: %s seconds TOTAL' % time_hhmmss)
