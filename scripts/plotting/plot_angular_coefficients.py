from utilities import logging, common
from utilities.io_tools import input_tools, output_tools
from utilities.styles import styles
from wremnants import theory_tools, plot_tools

import numpy as np
import mplhep as hep
import h5py


if __name__ == '__main__':
    parser = common.plot_parser()
    parser.add_argument("moments", help="Moments file `w_z_moments.hdf` with coefficients produced in w_z_gen_dists.py histmaker")
    parser.add_argument("--process", default="Z", choices=["Z", "W"], help="Process to be plotted")

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    with h5py.File(args.moments, "r") as ff:
        out = input_tools.load_results_h5py(ff)
    hhelicity = out[args.process]
    hhelicity = hhelicity[{"muRfact":1j, "muFfact":1j, "chargeVgen":0}]

    for var in ("absYVGen", "ptVGen"):

        hhelicities_1D = hhelicity.project(var, "helicity")
    
        hcoeffs = theory_tools.moments_to_angular_coeffs(hhelicities_1D)

        for i in hcoeffs.axes["helicity"]:
            if i == -1:
                continue

            # idx = hcoeffs.axes["helicity"].index(0)

            hcoeff = hcoeffs[{"helicity":complex(0,i)}]

            fig, ax1 = plot_tools.figure(hcoeff, xlabel=styles.xlabels.get(var, var), ylabel=f"$\mathrm{{A}}_{i}$",
                grid=False, automatic_scale=False, width_scale=1.2, logy=False)    
        
            hep.histplot(
                hcoeff,
                histtype="step",
                color="black",
                label="Prediction",
                yerr=True,
                ax=ax1,
                zorder=3,
                density=False,
                binwnorm=None,
            )

            plot_tools.addLegend(ax1, ncols=2, text_size=12, loc="upper left")
            plot_tools.fix_axes(ax1, logy=False)

            scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
            hep.cms.label(ax=ax1, lumi=None, fontsize=20*args.scaleleg*scale, 
                label=args.cmsDecor, data=False)

            outfile = f"angular_coefficient_{i}"
            outfile += f"_{var}"
            if args.postfix:
                outfile += f"_{args.postfix}"
            plot_tools.save_pdf_and_png(outdir, outfile)
            plot_tools.write_index_and_log(outdir, outfile, args=args)

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)

