import h5py
import os

from utilities import common, logging
from utilities.io_tools.conversion_tools import fitresult_pois_to_hist

import pdb

parser = common.base_parser()
parser.add_argument("--observed", type=str, default=None, help="fitresult file with observed results")
parser.add_argument("--expected", type=str, default=None, help="fitresult file with expected results")
parser.add_argument("-o", "--outfolder", type=str, default="./", help="Output folder")
parser.add_argument("--outputFile", type=str, default="results_unfolded", help="Output file name")
parser.add_argument("--override", action="store_true", help="Override output file if it exists")
parser.add_argument("--h5py", action="store_true", help="Dump output into hdf5 file using narf, use pickle by default")
args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

if not args.observed and not args.expected:
    raise IOError(f"Result from expected or observed fit must be specified with '--observed' or '--expected'")
result = {}
meta = None

if args.observed:
    result, meta = fitresult_pois_to_hist(args.observed.replace(".root",".hdf5"), result, uncertainties=None)
if args.expected:
    result, meta_exp = fitresult_pois_to_hist(args.expected.replace(".root",".hdf5"), result, uncertainties=None, expected=True)
    if not args.observed:
        meta = meta_exp
        meta_exp = None

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
        if meta is not None:
            ioutils.pickle_dump_h5py("meta", meta, f)
        if meta_exp is not None:
            ioutils.pickle_dump_h5py("meta_exp", meta_exp, f)
else:
    import pickle
    with open(outfile, "wb") as f:
        meta_info = {} 
        if meta is not None:
            meta_info["meta"] = meta
        if meta_exp is not None:
            meta_info["meta_exp"] = meta_exp
        pickle.dump({"results": result, **meta_info}, f)
