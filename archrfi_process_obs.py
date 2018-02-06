#!/usr/bin/env python
"""
Proces psrchive observations
@author: M Obrocka
python archrfi_process_obs.py --days 400 --localdir '/data' --maxsubint 180 --loglevel 'DEBUG'
"""
import argparse
import logging
import logging.handlers
from os import listdir
from os.path import join, isfile
from archrfi import ObsQuery, PulsarArchive, Timer

parser = argparse.ArgumentParser(
    description='Create Coast guard mask for psrchive timing observation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--days', dest='days', type=int, action='store', default=3,
    help='how far back to look for an observation in days')
parser.add_argument(
    '--localdir', dest='localdir', type=str, action='store', default='',
    help='path for the generated files')
parser.add_argument(
    '--maxsubint', dest='maxsubint', type=int, action='store', default=60,
    help='path for the logfile')
# parser.add_argument(
#     '--logfilename', dest='logfilename', type=str, action='store',
#     default='logging.out',
#     help='path for the logfile')
parser.add_argument(
    '--loglevel', dest='log_level', action='store', default='',
    help='log level to use, default None, options INFO, DEBUG, ERROR')
parser.add_argument(
    '--figures', dest='figures', type=bool, action='store', default=False,
    help='Save figures?')
args = parser.parse_args()

f = logging.Formatter(fmt='%(levelname)s:%(name)s: %(message)s '
                          '(%(asctime)s; %(filename)s:%(lineno)d)',
                      datefmt="%Y-%m-%d %H:%M:%S")

log_level = args.log_level.strip()
logging.basicConfig(level=eval('logging.%s' % log_level))
my_timer = Timer()

try:
    if args.figures:
        plotfig = args.figures
    else:
        plotfig = False
except:
    print('Skipping plotting!')

product_list = ['PulsarTimingArchiveProduct', 'PTUSETimingArchiveProduct']
for p in product_list:
    obs_query = ObsQuery(p)
    search = obs_query.paged_search(args.days)

    allfiles = [f for f in listdir(args.localdir) if isfile(join(args.localdir, f))]
    h5files = [f for f in allfiles if f.split('.')[-1] == 'h5']
    h5set = set(h5files)
    idx = 0

    for res in search:
        for obs in res.docs:
            archive = PulsarArchive(obs, idx)
            if (archive.archive_name + '.h5') in h5set:
                pass
            else:
                archive.download_observation(args.localdir, args.maxsubint)
                if archive.skip:
                    pass
                else:
                    archive.download_header()
                    archive.process_header()
                    archive.gen_archive()
                    before_bp = archive.estimate_bandpass()
                    weights = archive.run_coast_guard(cleaner_type="surgical")
                    mask = archive.run_coast_guard(cleaner_type="rcvrstd")
                    after_bp = archive.estimate_bandpass()
                    archive.count_rfi(mask)
                    if plotfig:
                        fig = archive.plot_mask(mask, save=True, directory=args.localdir)
                        fig.clear()
                        fig = archive.plot_bandpass(before_bp, save=True, directory=args.localdir)
                        fig.clear()
                    archive.write_h5(mask, before_bp, after_bp, directory=args.localdir)
                    archive.cleanup()
            idx += 1

time_hhmmss = my_timer.get_time_hhmmss()
my_timer.print_time(time_hhmmss)
print('Elapsed time: %s seconds TOTAL' % time_hhmmss)
