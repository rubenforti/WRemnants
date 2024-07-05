from utilities import logging, common
from utilities.io_tools import input_tools, output_tools
from utilities.styles import styles
from wremnants import theory_tools, plot_tools

import numpy as np
import matplotlib as mpl
import hist
import math
import mplhep as hep
import h5py

# def load_moments(filename, process="Z"):

#     return moments


    

#     if 'muRfact' in corrh.axes.name:
#         corrh = corrh[{'muRfact' : 1.j,}]
#     if 'muFfact' in corrh.axes.name:
#         corrh = corrh[{'muFfact' : 1.j,}]
    
#     # axes_names = ['massVgen','absYVgen','ptVgen','chargeVgen', 'helicity']
#     # if not list(corrh.axes.name) == axes_names:
#     #     raise ValueError (f"Axes [{corrh.axes.name}] are not the ones this functions expects ({axes_names})")
    
#     if np.count_nonzero(corrh[{"helicity" : -1.j}] == 0):
#         logger.warning("Zeros in sigma UL for the angular coefficients will give undefined behaviour!")
#     # histogram has to be without errors to load the tensor directly
#     corrh_noerrs = hist.Hist(*corrh.axes, storage=hist.storage.Double())
#     corrh_noerrs.values(flow=True)[...] = corrh.values(flow=True)

#     return corrh_noerrs

if __name__ == '__main__':
    parser = common.plot_parser()
    parser.add_argument("moments", help="Moments file `w_z_moments.hdf` with coefficients produced in w_z_gen_dists.py histmaker")
    parser.add_argument("--process", default="Z", choices=["Z", "W"], help="Process to be plotted")

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    if args.moments is not None:
        # raise NotImplementedError("Using moments file from histmaker output is not yet supported")
        with h5py.File(args.moments, "r") as ff:
            out = input_tools.load_results_h5py(ff)
        hhelicity = out[args.process]

        hcoefficients = theory_tools.moments_to_angular_coeffs(hhelicity)

        import pdb
        pdb.set_trace()

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)


    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)

