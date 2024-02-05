from utilities import logging, common
from utilities.io_tools import output_tools
from utilities.styles import styles
from utilities import boostHistHelpers as hh

from wremnants import plot_tools
from wremnants import histselections as sel
from wremnants.datasets.datagroups import Datagroups

import hist
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import mplhep as hep

import pdb

parser = common.plot_parser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histograms")
parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
parser.add_argument("--procFilters", type=str, nargs="*", default=["Fake", "QCD"], help="Filter to plot (default no filter, only specify if you want a subset")
parser.add_argument("--vars", type=str, nargs='*', default=["eta","pt","charge"], help="Variables to be considered in rebinning")
parser.add_argument("--rebin", type=int, nargs='*', default=[], help="Rebin axis by this value (default, 1, does nothing)")
parser.add_argument("--absval", type=int, nargs='*', default=[], help="Take absolute value of axis if 1 (default, 0, does nothing)")
parser.add_argument("--axlim", type=float, default=[], nargs='*', help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)")

parser.add_argument("--xAxisName", type=str, help="Name of x-axis for ABCD method", default="mt")
parser.add_argument("--xBinsLow", type=int, nargs='*', help="Binning of x-axis for ABCD method", default=[0,2,5,9,14,20,27,40])
parser.add_argument("--xBinsHigh", type=int, nargs='*', help="Binning of x-axis for ABCD method", default=[55,100])
parser.add_argument("--xOrder", type=int, default=2, help="Order in x-axis for fakerate parameterization")

args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)


outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)

groups = Datagroups(args.infile, excludeGroups=None, extendedABCD=True, integrateHighMT=True)

if args.axlim or args.rebin or args.absval:
    groups.set_rebin_action(args.vars, args.axlim, args.rebin, args.absval, rename=False)

abseta = "eta" in args.vars and len(args.absval) > args.vars.index("eta") and args.absval[args.vars.index("eta")]

logger.info(f"Load fakes")
groups.loadHistsForDatagroups(args.baseName, syst="", procsToRead=args.procFilters, applySelection=False)

histInfo = groups.getDatagroups()

### plot chi2 of fakerate fits
def plot_chi2(values, ndf=None, suffix="", outfile="chi2", xlim=None):
    
    if xlim is None:
        # mean, where outlier with sigma > 1 are rejected
        avg_chi2 = np.mean(values[abs(values - np.median(values)) < np.std(values)])
        xlim=(0, 2*avg_chi2)
    
    # set underflow and overflow
    values[values < xlim[0]] = xlim[0]
    values[values > xlim[1]] = xlim[1]

    fig, ax = plot_tools.figure(values, xlabel="$\chi^2$", ylabel="a.u.", cms_label="Work in progress", xlim=xlim, automatic_scale=False)#, ylim=ylim)

    ax.hist(values, bins=50, range=xlim, color="green", histtype="step", density=True, label="Fits")

    x = np.linspace(*xlim, 1000)
    chi_squared = stats.chi2.pdf(x, ndf)

    ax.plot(x, chi_squared, label=f'$\chi^2$ (ndf={ndf})', color="red")

    plot_tools.addLegend(ax, 1)

    if suffix:
        outfile = f"{suffix}_{outfile}"
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, 
        analysis_meta_info={"AnalysisOutput" : groups.getMetaInfo()},
        args=args,
    )


### plot fakerates vs. ABCD x-axis (e.g. mt)
def plot_xAxis(hs, params, covs, colors, labels, outdir, proc="X"):
    fit_threshold = args.xBinsLow[-1]
    ### plot the fit in each eta-pt-charge bin itself
    for idx_charge, charge_bins in enumerate(hs[0].axes["charge"]):
        logger.info(f"Make validation plots for charge index {idx_charge}")
        sign = r"\pm" if len(hs[0].axes["charge"])==1 else "-" if idx_charge==0 else "+" 

        for idx_eta, eta_bins in enumerate(hs[0].axes["eta"]):
            for idx_pt, pt_bins in enumerate(hs[0].axes["pt"]):
                ibin = {"charge":idx_charge ,"eta":idx_eta, "pt":idx_pt}

                ps = [p[idx_eta, idx_pt, idx_charge,:] for p in params]
                cs = [c[idx_eta, idx_pt, idx_charge,:,:] for c in covs]

                # fakerate factor as function of mt
                xfit = np.linspace(0,120,1000)

                fitFRF = [p[2,np.newaxis] * xfit**2 + p[1,np.newaxis] * xfit + p[0,np.newaxis] for p in ps]
                fitFRF_err = [(xfit**4*c[2,2,np.newaxis] + xfit**2*c[1,1,np.newaxis] + c[0,0,np.newaxis] \
                    + 2*xfit**3*c[2,1,np.newaxis] + 2*xfit**2*c[2,0,np.newaxis] + 2*xfit*c[1,0,np.newaxis])**0.5 for c in cs]

                # hBin = h[{**ibin, "passIso":slice(None,None,hist.sum)}]

                hPass = [h[{**ibin, "passIso":True}] for h in hs]
                hFail = [h[{**ibin, "passIso":False}] for h in hs]
                hFRF_bin = [hh.divideHists(hP, hF) for hP, hF in zip(hPass, hFail)]

                process= r"$\mathrm{"+proc+"}^{"+sign+r"}\rightarrow \mu^{"+sign+r"}\nu$"
                region="$"+str(round(pt_bins[0]))+"<p_\mathrm{T}<"+str(round(pt_bins[1]))\
                    +" ; "+str(round(eta_bins[0],1))+(r"<|\eta|<" if abseta else "<\eta<")+str(round(eta_bins[1],2))+"$"
                fits = ["$f(m_\mathrm{T}) = "+str(round(p[2]*2500.,2))+" * (m_\mathrm{T}/50)^2 "\
                    +("+" if p[1]>0 else "-")+str(round(abs(p[1])*50.,2))+" * (m_\mathrm{T}/50) "\
                    +("+" if p[0]>0 else "-")+str(round(abs(p[0]),1))+"$" for p in ps]

                ymin = min( 
                    min([min(f - e) for f,e in zip(fitFRF, fitFRF_err)]), 
                    min([min(h.values()-h.variances()**0.5) for h in hFRF_bin]))
                ymax = max( 
                    max([max(f + e) for f,e in zip(fitFRF, fitFRF_err)]), 
                    max([max(h.values()+h.variances()**0.5) for h in hFRF_bin]))
                ymax = min(5,ymax)
                ymin = max(0,ymin)
                ymax_line=ymax
                ymax += 0.2*(ymax-ymin)
                ylim = (ymin, ymax)

                fig, ax1 = plot_tools.figure(hFRF_bin[0], xlabel=styles.xlabels.get(args.xAxisName), ylabel="FRF",
                    cms_label="Work in progress",
                    automatic_scale=False, width_scale=1.2, ylim=ylim) 

                for i, (yf, ye) in enumerate(zip(fitFRF, fitFRF_err)):
                    ax1.fill_between(xfit, yf - ye, yf + ye, color=colors[i], alpha=0.2) 
                    ax1.plot(xfit, yf, color=colors[i])#, label="Fit")

                ax1.plot([fit_threshold,fit_threshold], [ymin, ymax_line], linestyle="--", color="black")

                ax1.text(1.0, 1.003, process, transform=ax1.transAxes, fontsize=30,
                        verticalalignment='bottom', horizontalalignment="right")
                ax1.text(0.03, 0.96, region, transform=ax1.transAxes, fontsize=20,
                        verticalalignment='top', horizontalalignment="left")
                for i, fit in enumerate(fits):
                    ax1.text(0.03, 0.90-0.06*i, fit, transform=ax1.transAxes, fontsize=20,
                            verticalalignment='top', horizontalalignment="left", color=colors[i])

                hep.histplot(
                    hFRF_bin,
                    histtype="errorbar",
                    color=colors,
                    label=labels,
                    linestyle="none",
                    stack=False,
                    ax=ax1,
                    yerr=True,
                )
                plot_tools.addLegend(ax1, 1)
                plot_tools.save_pdf_and_png(outdir, f"fit_charge{idx_charge}_eta{idx_eta}_pt{idx_pt}")


### plot fakerate parameters 1D
def plot1D(outfile, values, variances, var=None, x=None, xerr=None, label=None, xlim=None, bins=None, line=None, outdir="",
    ylabel="", postfix="", args={}, avg=None, avg_err=None, edges=None
):
    y = values
    yerr = np.sqrt(variances)

    ylim=(min(y-yerr), max(y+yerr))
    
    fig, ax = plot_tools.figure(values, xlabel=label, ylabel=ylabel, cms_label="Work in progress", xlim=xlim, ylim=ylim)
    if line is not None:
        ax.plot(xlim, [line,line], linestyle="--", color="grey", marker="")

    if avg is not None:
        ax.stairs(avg, edges, color="blue", label=f"${var}$ integrated")
        ax.bar(x=edges[:-1], height=2*avg_err, bottom=avg - avg_err, width=np.diff(edges), align='edge', linewidth=0, alpha=0.3, color="blue", zorder=-1)

    ax.errorbar(x, values, xerr=xerr, yerr=yerr, marker="", linestyle='none', color="k", label=f"${round(bins[0],1)} < {var} < {round(bins[1],1)}$")

    plot_tools.addLegend(ax, 1)

    if postfix:
        outfile += f"_{postfix}"
    plot_tools.save_pdf_and_png(outdir, outfile)

def plot_params_1D(h, params, outdir, pName, ip):
    xlabel = styles.xlabels[h.axes.name[0]]
    ylabel = styles.xlabels[h.axes.name[1]]
    xedges = np.squeeze(h.axes.edges[0])
    yedges = np.squeeze(h.axes.edges[1])

    h_pt = hLowMT[{"eta": slice(None,None,hist.sum)}]
    params_pt, cov_pt = sel.compute_fakerate1D(h_pt, args.xAxisName, True, use_weights=True, order=args.xOrder)

    h_eta = hLowMT[{"pt": slice(None,None,hist.sum)}]
    params_eta, cov_eta = sel.compute_fakerate1D(h_eta, args.xAxisName, True, use_weights=True, order=args.xOrder)

    for idx_charge, charge_bins in enumerate(hLowMT.axes["charge"]):
        logger.info(f"Make 1D parameter plots for charge index {idx_charge}")
        
        var = "|\eta|" if abseta else "\eta"
        label = styles.xlabels[h.axes.name[1]]           
        x = yedges[:-1]+(yedges[1:]-yedges[:-1])/2.
        xerr = (yedges[1:]-yedges[:-1])/2
        xlim=(min(yedges),max(yedges))

        avg = params_pt[:,idx_charge,ip]
        avg_err = cov_pt[:,idx_charge,ip,ip]**0.5
        outdir_eta = output_tools.make_plot_dir(outdir, f"{pName}_eta")
        for idx_eta, eta_bins in enumerate(h.axes["eta"]):
            plot1D(f"{pName}_charge{idx_charge}_eta{idx_eta}", values=params[idx_eta,:,idx_charge,ip], variances=cov[idx_eta,:,idx_charge,ip,ip], 
                ylabel=pName, line=0,
                var=var, x=x, xerr=xerr, label=label, xlim=xlim, bins=eta_bins, outdir=outdir_eta, avg=avg, avg_err=avg_err, edges=yedges)

        var = "p_\mathrm{T}"
        label = styles.xlabels[h.axes.name[0]]
        x = xedges[:-1]+(xedges[1:]-xedges[:-1])/2.
        xerr = (xedges[1:]-xedges[:-1])/2
        xlim=(min(xedges),max(xedges))  

        avg = params_eta[:,idx_charge,ip]
        avg_err = cov_eta[:,idx_charge,ip,ip]**0.5
        outdir_pt = output_tools.make_plot_dir(outdir, f"{pName}_pt")
        for idx_pt, pt_bins in enumerate(h.axes["pt"]):
            plot1D(f"{pName}_charge{idx_charge}_pt{idx_pt}", values=params[:,idx_pt,idx_charge,ip], variances=cov[:,idx_pt,idx_charge,ip,ip], 
                ylabel=pName, line=0,
                var=var, x=x, xerr=xerr, label=label, xlim=xlim, bins=pt_bins, outdir=outdir_pt, avg=avg, avg_err=avg_err, edges=xedges)



### plot fakerate parameters 2D
def plot_params_2D(h, values, outdir, suffix="", outfile="param", **kwargs):
    xlabel = styles.xlabels[h.axes.name[0]]
    ylabel = styles.xlabels[h.axes.name[1]]
    xedges = np.squeeze(h.axes.edges[0])
    yedges = np.squeeze(h.axes.edges[1])

    fig = plot_tools.makePlot2D(values=values, xlabel=xlabel, ylabel=ylabel, xedges=xedges, yedges=yedges, cms_label="WIP", **kwargs)
    if suffix:
        outfile = f"{suffix}_{outfile}"
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)


# plot composition of variances
#     # plot2D(f"cov_offset_slope_{idx_charge}", values=cov[...,idx,0,1], plot_title="Cov. "+("(-)" if idx==0 else "(+)"))
#     # plot2D(f"cor_offset_slope_{idx_charge}", values=cov[...,idx,0,1]/(np.sqrt(cov[...,idx,0,0]) * np.sqrt(cov[...,idx,1,1])), plot_title="Corr. "+("(-)" if idx==0 else "(+)"))
#     # # different parts for fakes in D
#     # plot2D(f"variance_correlation_{idx}", values=var_d_offset[...,idx])
#     # plot2D(f"variance_application_{idx}", values=var_d_application[...,idx])
#     # # relative variances
#     # info = dict(postfix="relative",
#     #     plot_uncertainties=True, logz=True, zlim=(0.04, 20)
#     # )
#     # plot2D(f"variance_slope_{idx}", values=val_d.sum(axis=-1)[...,idx], variances=var_d_slope[...,idx], plot_title="Rel. var. slope "+("(-)" if idx==0 else "(+)"), **info)
#     # plot2D(f"variance_offset_{idx}", values=val_d.sum(axis=-1)[...,idx], variances=var_d_offset[...,idx], plot_title="Rel. var. offset "+("(-)" if idx==0 else "(+)"), **info)
#     # plot2D(f"variance_correlation_{idx}", values=val_d.sum(axis=-1)[...,idx], variances=var_d_correlation[...,idx]*-1, plot_title="-1*Rel. var. corr. "+("(-)" if idx==0 else "(+)"), **info)
#     # plot2D(f"variance_application_{idx}", values=val_d.sum(axis=-1)[...,idx], variances=var_d_application[...,idx], plot_title="Rel. var. app. "+("(-)" if idx==0 else "(+)"), **info)
#     # plot2D(f"variance_slope_correlation_{idx}", values=val_d.sum(axis=-1)[...,idx], variances=var_d_sc[...,idx], plot_title="Rel. var. slope+corr. "+("(-)" if idx==0 else "(+)"), **info)
#     # plot2D(f"variance_fakerate_{idx}", values=val_d.sum(axis=-1)[...,idx], variances=var_d_soc[...,idx], plot_title="Rel. var. fakerate. "+("(-)" if idx==0 else "(+)"), **info)


### now call all those functions
hists=[]
pars=[]
covs=[]
for proc in args.procFilters:
    h = histInfo[proc].hists[args.baseName]
    hLowMT = hh.rebinHist(h, args.xAxisName, args.xBinsLow)
    params, cov, chi2, ndf = sel.compute_fakerate1D(hLowMT, axis_name_mt=args.xAxisName, overflow_mt=True, use_weights=True, order=args.xOrder, auxiliary_info=True)
    # plot_chi2(chi2.ravel(), ndf=ndf, xlim=(0,10), suffix=proc)

    # outdir1D = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_1D_{proc}/")
    # outdir2D = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_2D_{proc}/")
    # for idx_charge, charge_bins in enumerate(hLowMT.axes["charge"]):
    #     logger.info(f"Make validation plots for charge index {idx_charge}")
    #     sign = "" if len(hLowMT.axes["charge"])==1 else "-" if idx_charge==0 else "+" 

    #     for ip, pName, pRange in ((0,"offset", None), (1,"slope", (-0.02, 0.02)), (2,"quad", (-0.0004, 0.0004))):
    #         if ip > args.xOrder:
    #             break
    #         logger.info(f"Make validation plots for parameter {pName}")

    #         plot_params_2D(hLowMT, params[...,idx_charge,ip], outdir=outdir2D, outfile=f"{pName}_{idx_charge}", 
    #             plot_title=pName+sign, zlim=pRange)

    #         plot_params_2D(hLowMT, np.sqrt(cov[...,idx_charge,ip,ip]), outdir=outdir2D, outfile=f"unc_{pName}_{idx_charge}", 
    #             plot_title="$\Delta$ "+pName+sign)
            
    #     plot_params_1D(hLowMT, params, outdir1D, pName, ip)

    outdirX = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_{args.xAxisName}_{proc}/")
    xBins=[*args.xBinsLow, *args.xBinsHigh]
    h = hh.rebinHist(h, args.xAxisName, xBins)
    plot_xAxis([h], [params], [cov], ["black"], [proc], outdirX, proc)
    hists.append(h)
    pars.append(params)
    covs.append(cov)

if len(args.procFilters)>1:
    # summary all hists together in fakerate vs ABCD x-axis plot
    colors=["black","red","blue","green"]
    outdir = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_{args.xAxisName}_"+"_".join(args.procFilters))
    plot_xAxis(hists, pars, covs, colors[:len(hists)], args.procFilters, outdir)


