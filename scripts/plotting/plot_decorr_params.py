import re
import uproot

from narf import ioutils

from utilities import logging, common
from utilities.io_tools import combinetf_input, output_tools
from utilities.styles import styles

from wremnants import plot_tools

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import chi2

import pdb


if __name__ == '__main__':
    parser = common.plot_parser()
    parser.add_argument("infile", help="Fitresult file from combinetf with decorrelated fit")
    parser.add_argument("--infileInclusive", type=str, default=None, help="Fitresult file from combinetf with inclusive fit")
    parser.add_argument("--poiType", type=str, default="nois", help="Parameter type")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, help="x-axis range of the plot")
    parser.add_argument("--axes", type=str, default=["charge", "eta"], help="Names of decorrelation axes")
    parser.add_argument("--absoluteParam", action="store_true", help="Show plot as a function of absolute value of parameter (default is difference to SM prediction)")
    parser.add_argument("--title", type=str, default=None, help="Add a title to the plot on the upper right")

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    fitresult = combinetf_input.get_fitresult(args.infile)
    meta = ioutils.pickle_load_h5py(fitresult["meta"])
    meta_info = meta["meta_info"]
    # lumi = sum([c["lumi"] for c in meta["channel_info"].values()])

    with uproot.open(f"{args.infile.replace('.hdf5','.root')}:fitresults") as utree:
        nll = utree['nllvalfull'].array(library="np")

    if args.infileInclusive:
        fInclusive = combinetf_input.get_fitresult(args.infileInclusive)
        dfInclusive = combinetf_input.read_impacts_pois(fInclusive, poi_type=args.poiType, group=True, uncertainties=["stat", "muonCalibration"])
        with uproot.open(f"{args.infileInclusive.replace('.hdf5','.root')}:fitresults") as utree:
            nll_inclusive = utree['nllvalfull'].array(library="np")

    df = combinetf_input.read_impacts_pois(fitresult, poi_type=args.poiType, group=True, uncertainties=["stat", "muonCalibration"])

    df["Params"] = df["Name"].apply(lambda x: x.split("_")[0])
    df["Parts"] = df["Name"].apply(lambda x: x.split("_")[1:-1])

    for param, df_p in df.groupby("Params"):
        logger.info(f"Make plot for {param}")

        if param is not None and "MeV" in param:
            xlabel = param.split("MeV")[0]
            if xlabel.startswith("massShift"):
                proc = xlabel.replace("massShift","")[0]
                xlabel = "$m_\mathrm{"+str(proc)+"}$ [MeV]"
                offset = 80354 if proc=="W" else 91187.6

            if xlabel.startswith("Width"):
                proc = xlabel.replace("Width","")[0]
                xlabel = "$\Gamma_\mathrm{"+str(proc)+"}$ [MeV]"
                offset= 2091.13 if proc=="W" else 2494.13

            scale = float(re.search(r'\d+(\.\d+)?', param.split("MeV")[0].replace("p",".")).group())
            if "Diff" in param:
                scale *= 2 # take diffs by 2 as up and down pull in opposite directions
        else:
            scale = 1
            offset=0
            xlabel = param

        if not args.absoluteParam or "Diff" in param:
            xlabel = "$\Delta "+xlabel[1:]
            offset=0

        df_p["Names"] = df_p["Name"].apply(lambda x: "".join([x.split("MeV")[-1].split("_")[0] for x in x.split("_decorr")]))

        ylabels = [styles.xlabels.get(v,v) for v in args.axes]

        df_p["yticks"] = df_p["Names"]*0

        axes = []
        for i, v in enumerate(args.axes):
            df_p[v] = df_p["Names"].apply(lambda x: combinetf_input.decode_poi_bin(x, v))
            if all(df_p[v].values==None):
                continue
            axes.append(v)
            df_p[v] = df_p[v].astype(int)

        if "eta" in axes:
            df_p["yticks"] = df_p["eta"].apply(lambda x: round((x-12)*0.2,1)).astype(str)+"<\eta<"+df_p["eta"].apply(lambda x: round((x-12)*0.2+0.2,1)).astype(str)
        if "charge" in axes:
            df_p["yticks"] = df_p.apply(lambda x: x["yticks"].replace("eta","eta^{+}") if x["charge"]==1 else x["yticks"].replace("eta","eta^{-}"), axis=1)

        df_p["yticks"] = df_p["yticks"].apply(lambda x: f"${x}$")

        df_p.sort_values(by=axes, ascending=False, inplace=True)

        xCenter=0

        val = df_p["value"].values * scale + offset
        err = df_p["err_total"].values * scale
        err_stat = df_p["err_stat"].values * scale
        err_cal = df_p["err_muonCalibration"].values * scale

        yticks = df_p["yticks"].values

        if args.xlim is None:
            xlim = min(val-err), max(val+err)
            xwidth = xlim[1] - xlim[0]
            xlim = -0.05*xwidth + xlim[0], 0.05*xwidth + xlim[1]
        else:
            xlim = args.xlim
        
        ylim = (-1*(args.infileInclusive!=None), len(df_p) + (args.infileInclusive!=None))

        y = np.arange(0,len(df))+0.5 + (args.infileInclusive!=None)
        fig, ax1 = plot_tools.figure(None, xlabel=xlabel, ylabel="",#", ".join(ylabels), 
            cms_label=args.cmsDecor, #lumi=lumi,
            grid=True, automatic_scale=False, width_scale=0.8, height=4+0.24*len(df_p), xlim=xlim, ylim=ylim)    

        if args.infileInclusive:
            if len(dfInclusive) > 1:
                logger.warning(f"Found {len(dfInclusive)} values from the inclusive fit but was expecting 1, take first value")
            elif len(dfInclusive) == 0:
                raise RuntimeError (f"Found 0 values from the inclusive fit but was expecting 1")

            central = dfInclusive["value"].values[0] * scale + offset
            c_err_stat = dfInclusive["err_stat"].values[0] * scale
            c_err_cal = dfInclusive["err_muonCalibration"].values[0] * scale
            c_err = dfInclusive["err_total"].values[0] * scale

            ax1.errorbar([central], [0.], xerr=c_err_cal, color='orange', linewidth=3, marker="", linestyle="")
            ax1.errorbar([central], [0.], xerr=c_err, color='black', marker="o", linestyle="")
            ax1.errorbar([central], [0.], xerr=c_err_stat, color='red', marker="", linestyle="")

            ndf = len(df_p)-1

            logger.info(f"nll_inclusive = {nll_inclusive}; nll = {nll}")

            chi2_stat = 2*(nll_inclusive - nll)[0] + ndf
            plt.text(0.7 if val[-1]<offset else 0.05, 0.64+0.004*len(df_p), f"$<\chi^2/\mathrm{{ndf}}>$ = {str(round(chi2_stat,1))}/{ndf}", 
                fontsize=20, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)

            p_value = 1 - chi2.cdf(chi2_stat, ndf)
            logger.info(f"ndf = {ndf}; Chi2 = {chi2_stat}; p-value={p_value}")

            ax1.fill_between([central-c_err, central+c_err], ylim[0], ylim[1], color='gray', alpha=0.4)
            ax1.fill_between([central-c_err_cal, central+c_err_cal], ylim[0], ylim[1], color='orange', alpha=0.8)
            ax1.fill_between([central-c_err_stat, central+c_err_stat], ylim[0], ylim[1], color='red', alpha=0.6)

            yticks = ["Inclusive", *yticks]
            ytickpositions = [0., *y]

        else:
            ytickpositions = y


        ax1.plot([offset, offset], ylim, linestyle="--", marker="none", color="black", label="MC input value")

        ax1.set_yticks(ytickpositions, labels=yticks)
        ax1.minorticks_off()

        ax1.errorbar(val, y, xerr=err_stat, color='red', marker="", linestyle="", label="Stat unc.", zorder=3)
        ax1.errorbar(val, y, xerr=err_cal, color='orange', marker="", linestyle="", linewidth=3, label="Calibration unc.", zorder=2)
        ax1.errorbar(val, y, xerr=err, color='black', marker="", linestyle="", label="Measurement", zorder=1)
        ax1.plot(val, y, color='black', marker="o", linestyle="", zorder=4) # point on top
        # ax1.plot(val, y, color='black', marker="o") # plot black points on top

        plot_tools.addLegend(ax1, ncols=1, text_size=16, loc="upper right" if val[-1]<offset else "upper left")
        # plot_tools.fix_axes(ax1, logy=args.logy)

        if args.title:
            ax1.text(1.0,1.005, args.title, fontsize=28, horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes)

        outfile=f"decorr_{param}"
        if args.postfix:
            outfile += f"_{args.postfix}"

        plot_tools.save_pdf_and_png(outdir, outfile)
        plot_tools.write_index_and_log(outdir, outfile, 
            analysis_meta_info={"AnalysisOutput" : meta_info},
            args=args,
        )

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
