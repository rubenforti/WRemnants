import h5py
import os
import narf

from utilities import common, logging
from utilities.io_tools.conversion_tools import fitresult_pois_to_hist
from utilities.io_tools import output_tools

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

res_dict = {
    "results" : result,
    "combine_meta" : meta,
    "meta_info" : narf.ioutils.make_meta_info_dict(args=args, wd=common.base_dir),
}

if args.h5py:
    from narf import ioutils
    with h5py.File(outfile, "w") as f:
        logger.debug(f"Pickle and dump results")
        for k,v in res_dict.items():
            ioutils.pickle_dump_h5py(k, v, f)
else:
    import pickle
    with open(outfile, "wb") as f:
        pickle.dump(res_dict, f)
