# CoastGuard RFI masks for timing observations (psrchive format)

Docker-based pipeline for pulsar timing observation in psrchive format that produces
'Coast Guard' binary mask.

### Prerequisites

Download psrchive-docker from https://github.com/pinsleepe/psrchive-docker.

To build:

    docker build -t psrchive-docker .

To run image with mounted data directory into the docker container with the -v flag:

    docker run -it -v <data_location>:/data psrchive-docker /bin/bash

### Running

Docker container is provided to run archrfi_process_obs.py script, which requires
psrchive package to process the timing observations. Once inside docker environment run:

    ./archrfi_process_obs.py --days 30 --localdir '/data' --maxsubint 60 --loglevel 'INFO'

HDF5 files will be saved to mounted <data_location>.

To run the reduction script rfi_reduction.py you don't need a docker container but you
will need Python 3 and following pakages:
- numpy
- pandas
- h5py
- astropy
- matplotlib
- calmap.

Then run:

    ./rfi_reduction.py --dest '<path_to_h5_files>' --plot True

### Authors

* **Monika Obrocka** - *Initial work* - [pinsleepe](https://github.com/pinsleepe)

### Acknowledgments

* M. Serylak for helping with CoastGuard set up

