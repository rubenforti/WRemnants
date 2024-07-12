from utilities import logging, common, boostHistHelpers as hh
from utilities.io_tools import input_tools, output_tools
from utilities.styles import styles
from wremnants import theory_tools, plot_tools

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import chi2

if __name__ == '__main__':
    parser = common.plot_parser()
    parser.add_argument("helicities", nargs="+", type=str, help="File `w_z_helicity_xsecs.hdf` with helicity cross sections produced in w_z_gen_dists.py histmaker")
    parser.add_argument("--labels", nargs="+", type=str, help="Labels for the different input files")
    parser.add_argument("--process", default="Z", choices=["Z", "W"], help="Process to be plotted")
    parser.add_argument("--plotXsecs", action="store_true", help="Plot helicity cross sections instead of angular coefficients")

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    plot_coefficients = not args.plotXsecs

    helicities = []
    helicity_sum = None
    weight_sum_all=0
    xsec=0
    for helicity_file in args.helicities:
        with h5py.File(helicity_file, "r") as ff:
            out = input_tools.load_results_h5py(ff)
        hhelicity = out[args.process]
        hhelicity = hhelicity[{"muRfact":1j, "muFfact":1j, "chargeVgen":0, "massVgen":0}]
        hhelicity = hh.disableFlow(hhelicity, "absYVGen")

        weight_sum=0
        with h5py.File(helicity_file.replace("helicity_xsecs","gen_dists"), "r") as ff:
            out = input_tools.load_results_h5py(ff)
            for key in out.keys():
                if key.startswith(args.process) and key not in ["meta_info"]:
                    xsec = out[key]["dataset"]["xsec"]
                    weight_sum = out[key]["weight_sum"]
                    weight_sum_all += weight_sum

        if len(args.helicities)>1:
            if helicity_sum is None:
                helicity_sum = hhelicity.copy()
            else:
                helicity_sum = hh.addHists(helicity_sum, hhelicity)

        hhelicity = hh.scaleHist(hhelicity, xsec/weight_sum)
        helicities.append(hhelicity)

    if helicity_sum is not None:
        # under the assumption that xsec is the same for all processes
        helicity_sum = hh.scaleHist(helicity_sum, xsec/weight_sum_all)

    if len(helicities) == 2:
        chi2_stat = np.sum( (helicities[0].values(flow=True)-helicities[1].values(flow=True))**2/(helicities[0].variances(flow=True)+helicities[1].variances(flow=True)) )
        dof = len(helicities[0].values(flow=True).flatten()) - 1
        p_val = chi2.sf(chi2_stat, dof)

        logger.info(f"Chi2/ndf = {chi2_stat} with p-value = {p_val}")

    colors = mpl.colormaps["tab10"]

    for var in ("absYVGen", "ptVGen"):

        hhelicities_1D = [h.project(var, "helicity") for h in helicities]
        if helicity_sum is not None:
            h1d_sum = helicity_sum.project(var, "helicity")

        if plot_coefficients:
            hists1d = [theory_tools.helicity_xsec_to_angular_coeffs(h) for h in hhelicities_1D]
            if helicity_sum is not None:
                h1d_sum = theory_tools.helicity_xsec_to_angular_coeffs(h1d_sum)
        else:
            hists1d = hhelicities_1D
            if helicity_sum is not None:
                h1d_sum = helicity_sum.project(var, "helicity")

        for i in hists1d[0].axes["helicity"]:
            if i == -1 and plot_coefficients:
                continue

            h1ds = [h[{"helicity":complex(0,i)}] for h in hists1d]

            ylabel = f"$\mathrm{{A}}_{i}$" if plot_coefficients else r"$\sigma_{"+(r"\mathrm{UL}" if i==-1 else str(i))+r"}\,[\mathrm{pb}]$"
            fig, ax1 = plot_tools.figure(h1ds[0], xlabel=styles.xlabels.get(var, var), 
                ylabel=ylabel,
                grid=False, automatic_scale=False, width_scale=1.2, logy=False)    
            
            y_min=np.inf
            y_max=-np.inf
            for j, h in enumerate(h1ds):
                hep.histplot(
                    h,
                    histtype="step",
                    color=colors(j),
                    label=args.labels[j],
                    yerr=False,
                    ax=ax1,
                    zorder=3,
                    density=False,
                    binwnorm=None if plot_coefficients else 1,
                )
                y_min = min(y_min, min(h.values()))
                y_max = max(y_max, max(h.values()))

            if helicity_sum is not None:
                h = h1d_sum[{"helicity":complex(0,i)}]
                hep.histplot(
                    h,
                    histtype="step",
                    color="black",
                    label="Sum",
                    linestyle="-",
                    yerr=False,
                    ax=ax1,
                    zorder=3,
                    density=False,
                    binwnorm=None if plot_coefficients else 1,
                )    
                y_min = min(y_min, min(h.values()))
                y_max = max(y_max, max(h.values()))

            if not plot_coefficients:
                y_min, y_max = ax1.get_ylim()
            yrange = y_max-y_min

            x_min, x_max = ax1.get_xlim()
            plt.plot([x_min, x_max], [0,0], color="black", linestyle="--")            

            ax1.set_ylim([y_min-yrange*0.1, y_max+yrange*0.2])

            plot_tools.addLegend(ax1, ncols=2, text_size=12, loc="upper left")
            plot_tools.fix_axes(ax1, logy=False)

            scale = max(1, np.divide(*ax1.get_figure().get_size_inches())*0.3)
            hep.cms.label(ax=ax1, lumi=None, fontsize=20*args.scaleleg*scale, 
                label=args.cmsDecor, data=False)

            if plot_coefficients:
                outfile = f"angular_coefficient_{i}"
            else:
                outfile = f"helicity_xsec_" + ("UL" if i==-1 else str(i))

            outfile += f"_{var}"
            if args.postfix:
                outfile += f"_{args.postfix}"
            plot_tools.save_pdf_and_png(outdir, outfile)
            plot_tools.write_index_and_log(outdir, outfile, args=args)

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
