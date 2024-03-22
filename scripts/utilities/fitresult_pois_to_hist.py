import h5py
import os

from utilities import common, logging
from utilities.io_tools.conversion_tools import fitresult_pois_to_hist

import pdb

parser = common.base_parser()
parser.add_argument("infile", type=str, help="fitresult file")
parser.add_argument("-o", "--outfolder", type=str, default="./", help="Output folder")
parser.add_argument("--outputFile", type=str, default="results_unfolded", help="Output file name")
parser.add_argument("--override", action="store_true", help="Override output file if it exists")
parser.add_argument("--h5py", action="store_true", help="Dump output into hdf5 file using narf, use pickle by default")
args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

# what pois are read and converted to histograms
infile = args.infile.replace(".root",".hdf5")

result, meta = fitresult_pois_to_hist(infile, uncertainties=None)

# saving the result histograms
outfile = f"{args.outfolder}/{args.outputFile}{ '.hdf5' if args.h5py else '.pkl'}"
if os.path.isfile(outfile) and not args.override:
    raise IOError(f"The file {outfile} already exists, use '--override' to override it")

if not os.path.exists(args.outfolder):
    logger.info(f"Creating output folder {args.outfolder}")
    os.makedirs(args.outfolder)

if args.h5py:
    from narf import ioutils
    with h5py.File(outfile, "w") as f:
        logger.debug(f"Pickle and dump results")
        ioutils.pickle_dump_h5py("results", result, f)
        ioutils.pickle_dump_h5py("meta", meta, f)
else:
    import pickle
    with open(outfile, "wb") as f:
        pickle.dump({"results": result, "meta": meta}, f)