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
import matplotlib as mpl
from matplotlib import pyplot as plt
import mplhep as hep
import uncertainties as unc
import uncertainties.unumpy as unp

import pdb


### plot chi2 of fakerate fits
def plot_chi2(values, ndf, outdir, suffix="", outfile="chi2", xlim=None):
    
    if xlim is None:
        # mean, where outlier with sigma > 1 are rejected
        avg_chi2 = np.mean(values[abs(values - np.median(values)) < np.std(values)])
        xlim=(0, 2*avg_chi2)
    
    # set underflow and overflow
    values[values < xlim[0]] = xlim[0]
    values[values > xlim[1]] = xlim[1]

    fig, ax = plot_tools.figure(values, xlabel="$\chi^2$", ylabel="a.u.", cms_label=args.cmsDecor, xlim=xlim, automatic_scale=False)#, ylim=ylim)

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
def plot_xAxis(hs, params, covs, frfs, colors, labels, outdir, proc="X"):
    fit_threshold = args.xBinsLow[-1]
    ### plot the fit in each eta-pt-charge bin itself
    for idx_charge, charge_bins in enumerate(hs[0].axes["charge"]):
        logger.info(f"Make validation plots for charge index {idx_charge}")
        sign = r"\pm" if len(hs[0].axes["charge"])==1 else "-" if idx_charge==0 else "+" 

        for idx_eta, eta_bins in enumerate(hs[0].axes["eta"]):
            for idx_pt, pt_bins in enumerate(hs[0].axes["pt"]):
                ibin = {"charge":idx_charge ,"eta":idx_eta, "pt":idx_pt}

                # fakerate factor as function of mt
                xfit = np.linspace(0,120,1000)

                if args.xAxisName2:
                    ps = [p[idx_eta, idx_charge,:] for p in params]
                    cs = [c[idx_eta, idx_charge,:,:] for c in covs]
                    xfits = [xfit, np.diff(pt_bins)/2+pt_bins[0]]
                else:
                    ps = [p[idx_eta, idx_pt, idx_charge,:] for p in params]
                    cs = [c[idx_eta, idx_pt, idx_charge,:,:] for c in covs]
                    xfits = [xfit]

                pcs = [np.array([unc.correlated_values(p, c)]) for p,c in zip(ps, cs)] # make error propagation
                fitFRF = [f(*xfits, p) for f,p in zip(frfs,pcs)]

                fitFRF_err = [np.array([a.s for a in np.squeeze(arr)]) for arr in fitFRF]
                fitFRF = [np.array([a.n for a in np.squeeze(arr)]) for arr in fitFRF]

                hPass = [h[{**ibin, "passIso":True}] for h in hs]
                hFail = [h[{**ibin, "passIso":False}] for h in hs]
                hFRF_bin = [hh.divideHists(hP, hF) for hP, hF in zip(hPass, hFail)]

                process= r"$\mathrm{"+proc+"}^{"+sign+r"}\rightarrow \mu^{"+sign+r"}\nu$"
                region="$"+str(round(pt_bins[0]))+"<p_\mathrm{T}<"+str(round(pt_bins[1]))\
                    +" ; "+str(round(eta_bins[0],1))+(r"<|\eta|<" if abseta else "<\eta<")+str(round(eta_bins[1],2))+"$"

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
                    cms_label=args.cmsDecor, automatic_scale=False, width_scale=1.2, ylim=ylim) 

                for i, (yf, ye) in enumerate(zip(fitFRF, fitFRF_err)):
                    ax1.fill_between(xfit, yf - ye, yf + ye, color=colors[i], alpha=0.2) 
                    ax1.plot(xfit, yf, color=colors[i])#, label="Fit")

                ax1.plot([fit_threshold,fit_threshold], [ymin, ymax_line], linestyle="--", color="black")

                ax1.text(1.0, 1.003, process, transform=ax1.transAxes, fontsize=30,
                        verticalalignment='bottom', horizontalalignment="right")
                ax1.text(0.03, 0.96, region, transform=ax1.transAxes, fontsize=20,
                        verticalalignment='top', horizontalalignment="left")

                for i, (p, c) in enumerate(zip(ps, cs)):
                    if args.xAxisName2:
                        idx=0
                        chars=["a","b","c","d","e"]

                        fit = "$f(m_\mathrm{T}, p_\mathrm{T}) = a(p_\mathrm{T})"
                        if args.xOrder>=1:
                            fit += "+ b(p_\mathrm{T}) \cdot m_\mathrm{T}"
                        if args.xOrder>=2:
                            fit += "+ c(p_\mathrm{T}) \cdot m_\mathrm{T}^2"
                        fit +="$"
                        ax1.text(0.03, 0.90-0.06*i, fit, transform=ax1.transAxes, fontsize=20,
                            verticalalignment='top', horizontalalignment="left", color=colors[i])

                        factor1=1
                        for n in range(args.xOrder+1):
                            factor2=1
                            for m in range(args.xOrder2[n]+1):
                                factor=factor1*factor2
                                ax1.text(1.03+0.2*i, 0.84-0.06*idx, f"${chars[n]}_{m}={round(p[idx]*factor,2)}\pm{round(c[idx,idx]**0.5*factor,2)}$", 
                                    transform=ax1.transAxes, fontsize=20,
                                    verticalalignment='top', horizontalalignment="left", color=colors[i])
                                factor2 *= 50
                                idx+=1
                            factor1 *= 50

                    else:
                        fit = "$f(m_\mathrm{T}) = "+str(round(p[2]*2500.,2))+" * (m_\mathrm{T}/50)^2 "\
                            +("+" if p[1]>0 else "-")+str(round(abs(p[1])*50.,2))+" * (m_\mathrm{T}/50) "\
                            +("+" if p[0]>0 else "-")+str(round(abs(p[0]),1))+"$"

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
    ylabel="", postfix="", avg=None, avg_err=None, edges=None
):
    y = values
    yerr = np.sqrt(variances)

    ylim=(min(y-yerr), max(y+yerr))
    
    fig, ax = plot_tools.figure(values, xlabel=label, ylabel=ylabel, cms_label=args.cmsDecor, xlim=xlim, ylim=ylim)
    if line is not None:
        ax.plot(xlim, [line,line], linestyle="--", color="grey", marker="")

    if avg is not None:
        ax.stairs(avg, edges, color="blue", label=f"${var}$ integrated")
        ax.bar(x=edges[:-1], height=2*avg_err, bottom=avg - avg_err, width=np.diff(edges), align='edge', linewidth=0, alpha=0.3, color="blue", zorder=-1)

    if bins:
        binlabel=f"${round(bins[0],1)} < {var} < {round(bins[1],1)}$"
    else:
        binlabel="Inclusive"

    ax.errorbar(x, values, xerr=xerr, yerr=yerr, marker="", linestyle='none', color="k", label=binlabel)

    plot_tools.addLegend(ax, 1)

    if postfix:
        outfile += f"_{postfix}"
    plot_tools.save_pdf_and_png(outdir, outfile)

def plot_params_1D(h, params, outdir, pName, ip, idx_charge, charge_bins):
    xlabel = styles.xlabels[h.axes.name[0]]
    ylabel = styles.xlabels[h.axes.name[1]]
    xedges = np.squeeze(h.axes.edges[0])
    yedges = np.squeeze(h.axes.edges[1])

    h_pt = hLowMT[{"eta": slice(None,None,hist.sum)}]
    params_pt, cov_pt, _ = sel.compute_fakerate(h_pt, args.xAxisName, True, use_weights=True, order=args.xOrder)

    h_eta = hLowMT[{"pt": slice(None,None,hist.sum)}]
    params_eta, cov_eta, _ = sel.compute_fakerate(h_eta, args.xAxisName, True, use_weights=True, order=args.xOrder)

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

    fig = plot_tools.makePlot2D(values=values, xlabel=xlabel, ylabel=ylabel, xedges=xedges, yedges=yedges, cms_label=args.cmsDecor, **kwargs)
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

### plot closure
def plot_closure(h, outdir, suffix="", outfile=f"closureABCD", ratio=True, proc="", ylabel="a.u."):
    fakerate_integration_axes = [a for a in ["eta","pt","charge"] if a not in args.vars]
    threshold = args.xBinsLow[-1]

    info=dict(fakerate_axis=args.xAxisName, fakerate_bins=args.xBinsLow,
        fakerate_integration_axes=fakerate_integration_axes, integrate=True, axis_name_2=args.xAxisName2, order_2=args.xOrder2
    )

    hD = sel.fakeHistABCD(h, thresholdMT=threshold, fakerate_integration_axes=fakerate_integration_axes, integrateHighMT=True)
    hD_pol0 = sel.fakeHistExtendedABCD(h, order=0, **info)
    hD_pol1 = sel.fakeHistExtendedABCD(h, order=1, **info)
    hD_pol2 = sel.fakeHistExtendedABCD(h, order=2, **info)
    hD_sig = sel.signalHistWmass(h, thresholdMT=threshold, integrateHighMT=True)


    hss = [hD_sig, hD, hD_pol0, hD_pol1, hD_pol2]
    labels = ["D", "C*B/A", "C*B/A pol0", "C*B/A pol1", "C*B/A pol2"]
    linestyles = ["-", "-", "--", ":", "dashdot"]
    colors = mpl.colormaps["tab10"]
    
    if "charge" in hss[0].axes.name and len(hss[0].axes["charge"])==1:
        hss = [h[{"charge":slice(None,None,hist.sum)}] for h in hss]

    axes = hss[0].axes.name

    if len(axes)>1:
        hss = [sel.unrolledHist(h, obs=axes[::-1]) for h in hss]

    hs = hss
    hr = [hh.divideHists(h1d, hss[0]) for h1d in hss]

    ymin = 0
    ymax = max([h.values(flow=False).max() for h in hs])
    yrange = ymax - ymin
    ymin = ymin if ymin == 0 else ymin - yrange*0.3
    ymax = ymax + yrange*0.3

    xlabel = styles.xlabels[axes[0]] if len(axes)==1 and axes in styles.xlabels else f"{'-'.join(axes)} Bin"
    if ratio:
        fig, ax1, ax2 = plot_tools.figureWithRatio(hss[0], xlabel=xlabel, ylabel=ylabel, cms_label=args.cmsDecor,
                                             rlabel=f"1/{labels[0]}", rrange=(0.25, 1.75), 
                                             automatic_scale=False, width_scale=1.2, ylim=(ymin, ymax))
    else:
        fig, ax1 = plot_tools.figure(hss[0], xlabel=xlabel, ylabel=ylabel, cms_label=args.cmsDecor,
                                             automatic_scale=False, width_scale=1.2, ylim=(ymin, ymax))        

    hep.histplot(
        hs,
        histtype = "step",
        color = [colors(i) for i in range(len(hss))],
        label = labels[:len(hs)],
        linestyle = linestyles[:len(hs)],
        ax = ax1
    )
    
    if ratio:
        hep.histplot(
            hr,
            yerr=True,
            histtype = "step",
            color = [colors(i) for i in range(len(hr))],
            label = labels[:len(hr)],
            linestyle = linestyles[:len(hr)],
            ax = ax2
        )
    #     ax.fill_between(x, y - err, y + err, alpha=0.3, color=colors(i), step="post")

    ax1.text(1.0, 1.003, styles.process_labels[proc], transform=ax1.transAxes, fontsize=30,
            verticalalignment='bottom', horizontalalignment="right")
    plot_tools.addLegend(ax1, ncols=2, text_size=12)
    plot_tools.fix_axes(ax1, ax2)

    if suffix:
        outfile = f"{suffix}_{outfile}"
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)


if __name__ == '__main__':
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
    parser.add_argument("--xBinsHigh", type=int, nargs='*', help="Binning of x-axis for ABCD method", default=[50,70,120])
    parser.add_argument("--xOrder", type=int, default=2, help="Order in x-axis for fakerate parameterization")

    parser.add_argument("--xAxisName2", type=str, help="Name of second axis for ABCD method, 'None' means 1D fakerates", default=None)
    parser.add_argument("--xOrder2", type=int, nargs='*', default=[2,1,0], help="Order in second axis for fakerate parameterization")

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

    hists=[]
    pars=[]
    covs=[]
    frfs=[]
    for proc in args.procFilters:
        h = histInfo[proc].hists[args.baseName]

        if proc != groups.fakeName:
            plot_closure(h, outdir, suffix=proc, proc=proc)

        hLowMT = hh.rebinHist(h, args.xAxisName, args.xBinsLow)
        params, cov, frf, chi2, ndf = sel.compute_fakerate(hLowMT, axis_name=args.xAxisName, overflow=True, use_weights=True, 
            order=args.xOrder, axis_name_2=args.xAxisName2, order_2=args.xOrder2, auxiliary_info=True)
        plot_chi2(chi2.ravel(), ndf, outdir, xlim=(0,10), suffix=proc)

        outdir1D = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_1D_{proc}/")
        outdir2D = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_2D_{proc}/")
        for idx_charge, charge_bins in enumerate(hLowMT.axes["charge"]):
            logger.info(f"Make parameter validation plots for charge index {idx_charge}")

            sign = "" if len(hLowMT.axes["charge"])==1 else "-" if idx_charge==0 else "+" 
            
            if args.xAxisName2:
                idx_p=0
                outdir_p = output_tools.make_plot_dir(outdir, f"params")
                for ip, (pName, pRange) in enumerate([("offset", None), ("slope", (-0.02, 0.02)), ("quad", (-0.0004, 0.0004))]):
                    if ip > args.xOrder:
                        break
                    for ip2, (pName2, pRange2) in enumerate([("offset", None), ("slope", (-0.02, 0.02)), ("quad", (-0.0004, 0.0004))]):
                        if ip2 > args.xOrder2[ip]:
                            break

                        xedges = np.squeeze(h.axes.edges[0])

                        var = "|\eta|" if abseta else "\eta"
                        label = styles.xlabels[h.axes.name[0]]           
                        x = xedges[:-1]+(xedges[1:]-xedges[:-1])/2.
                        xerr = (xedges[1:]-xedges[:-1])/2
                        xlim=(min(xedges),max(xedges))

                        plot1D(f"{pName}_{pName2}_charge{idx_charge}", values=params[...,idx_charge,idx_p], variances=cov[...,idx_charge,idx_p,idx_p], 
                            ylabel=f"{pName}-{pName2}", line=0,
                            var=var, x=x, xerr=xerr, label=label, xlim=xlim, outdir=outdir_p, edges=xedges)

                        idx_p+=1
            else:
                for ip, (pName, pRange) in enumerate([("offset", None), ("slope", (-0.02, 0.02)), ("quad", (-0.0004, 0.0004))]):
                    plot_params_2D(hLowMT, params[...,idx_charge,ip], outdir=outdir2D, outfile=f"{pName}_{idx_charge}", 
                        plot_title=pName+sign, zlim=pRange)
                    plot_params_2D(hLowMT, np.sqrt(cov[...,idx_charge,ip,ip]), outdir=outdir2D, outfile=f"unc_{pName}_{idx_charge}", 
                        plot_title="$\Delta$ "+pName+sign)
                    plot_params_1D(hLowMT, params, outdir1D, pName, ip, idx_charge, charge_bins)
            
        outdirX = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_{args.xAxisName}_{proc}/")
        xBins=[*args.xBinsLow, *args.xBinsHigh]
        h = hh.rebinHist(h, args.xAxisName, xBins)
        plot_xAxis([h], [params], [cov], [frf], ["black"], [proc], outdirX, proc)
        hists.append(h)
        pars.append(params)
        covs.append(cov)
        frfs.append(frf)

    if len(hists)>1:
        # summary all hists together in fakerate vs ABCD x-axis plot
        colors=["black","red","blue","green"]
        outdir = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_{args.xAxisName}_"+"_".join(args.procFilters))
        plot_xAxis(hists, pars, covs, frfs, colors[:len(hists)], args.procFilters, outdir)


