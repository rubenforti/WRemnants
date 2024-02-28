from utilities import logging, common
from utilities.io_tools import output_tools
from utilities.styles import styles
from utilities import boostHistHelpers as hh

from wremnants import plot_tools
from wremnants.datasets.datagroups import Datagroups
from wremnants import histselections as sel

import hist
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import pyplot as plt
import mplhep as hep
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

import itertools

import pdb

def exp_fall(x, a, b, c):
    return a * np.exp(-b * x) + c

def exp_fall_unc(x, a, b, c):
    return a * unp.exp(-b * x) + c

### plot chi2 of fakerate fits
def plot_chi2(values, ndf, outdir, suffix="", outfile="chi2", xlim=None):
    
    if xlim is None:
        # mean, where outlier with sigma > 1 are rejected
        avg_chi2 = np.mean(values[abs(values - np.median(values)) < np.std(values)])
        xlim=(0, 2*avg_chi2)
    
    # set underflow and overflow
    values = values.copy()
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

                if args.xsmoothingAxisName:
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
                    if args.xsmoothingAxisName:
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
                            for m in range(args.xsmoothingOrder[n]+1):
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

# plot fits of QCD shapes in sideband regions for full extended ABCD method
def plot_abcd_fits(h, outdir):
    # define sideband ranges
    nameX, nameY = sel.get_abcd_axes(h)
    binsX = h.axes[nameX].edges
    binsY = h.axes[nameY].edges

    dx = slice(complex(0, binsX[1]), complex(0, binsX[2]), hist.sum)
    dy = slice(complex(0, binsY[1]), complex(0, binsY[2]), hist.sum)

    s = hist.tag.Slicer()
    if nameX in ["mt"]:
        x = slice(complex(0, binsX[2]), None, hist.sum)
        d2x = slice(complex(0, binsX[0]), complex(0, binsX[1]), hist.sum)
    elif nameX in ["dxy", "iso"]:
        x = slice(complex(0, binsX[0]), complex(0, binsX[1]), hist.sum)
        d2x = slice(complex(0, binsX[2]), None, hist.sum)

    if nameY in ["mt"]:
        y = slice(complex(0, binsY[2]), None, hist.sum)
        d2y = slice(complex(0, binsY[0]), complex(0, binsY[1]), hist.sum)
    elif nameY in ["dxy", "iso"]:
        y = slice(complex(0, binsY[0]), complex(0, binsY[1]), hist.sum)
        d2y = slice(complex(0, binsY[2]), None, hist.sum)

    flow=True
    logy=False
    h_pt = h.project("pt")
    bincenters = h_pt.axes["pt"].centers
    binwidths = np.diff(h_pt.axes["pt"].edges)/2.
    etavar="|\eta|"
    for idx_charge, charge_bins in enumerate(h.axes["charge"]):
        for idx_eta, eta_bins in enumerate(h.axes["eta"]):
            # select sideband regions
            a = h[{"charge":idx_charge, "eta":idx_eta, nameX: dx, nameY: dy}].values(flow=flow)
            b = h[{"charge":idx_charge, "eta":idx_eta, nameX: dx, nameY: y}].values(flow=flow)
            c = h[{"charge":idx_charge, "eta":idx_eta, nameX: x, nameY: dy}].values(flow=flow)
            bx = h[{"charge":idx_charge, "eta":idx_eta, nameX: d2x, nameY: y}].values(flow=flow)
            cy = h[{"charge":idx_charge, "eta":idx_eta, nameX: x, nameY: d2y}].values(flow=flow)
            ax = h[{"charge":idx_charge, "eta":idx_eta, nameX: d2x, nameY: dy}].values(flow=flow)
            ay = h[{"charge":idx_charge, "eta":idx_eta, nameX: dx, nameY: d2y}].values(flow=flow)
            axy = h[{"charge":idx_charge, "eta":idx_eta, nameX: d2x, nameY: d2y}].values(flow=flow)

            avar = h[{"charge":idx_charge, "eta":idx_eta, nameX: dx, nameY: dy}].variances(flow=flow)
            bvar = h[{"charge":idx_charge, "eta":idx_eta, nameX: dx, nameY: y}].variances(flow=flow)
            cvar = h[{"charge":idx_charge, "eta":idx_eta, nameX: x, nameY: dy}].variances(flow=flow)
            bxvar = h[{"charge":idx_charge, "eta":idx_eta, nameX: d2x, nameY: y}].variances(flow=flow)
            cyvar = h[{"charge":idx_charge, "eta":idx_eta, nameX: x, nameY: d2y}].variances(flow=flow)
            axvar = h[{"charge":idx_charge, "eta":idx_eta, nameX: d2x, nameY: dy}].variances(flow=flow)
            ayvar = h[{"charge":idx_charge, "eta":idx_eta, nameX: dx, nameY: d2y}].variances(flow=flow)
            axyvar = h[{"charge":idx_charge, "eta":idx_eta, nameX: d2x, nameY: d2y}].variances(flow=flow)

            for n, values, values_var in (
                ("a", a, avar),
                ("b", b, bvar),
                ("c", c, cvar),
                ("bx", bx, bxvar),
                ("cy", cy, cyvar),
                ("ax", ax, axvar),
                ("ay", ay, ayvar),
                ("axy", axy, axyvar),
            ):      
                values_err = values_var**0.5
                xlim = [min(bincenters-binwidths), max(bincenters+binwidths)]
                if logy:
                    ylim = [0.1, max(values+values_err)*2]
                else:
                    ylim = [0, max(values+values_err)*1.1]
                
                fig, ax = plot_tools.figure(h_pt, ylabel="Events/bin", xlabel="$p_\mathrm{T}$ (GeV)", cms_label=args.cmsDecor, xlim=xlim, ylim=ylim, logy=logy)

                binlabel=f"${round(eta_bins[0],1)} < {etavar} < {round(eta_bins[1],1)}$"

                ax.errorbar(bincenters, values, xerr=binwidths, yerr=values_err, marker="", linestyle='none', color="k", label=binlabel)

                params, cov = curve_fit(exp_fall, bincenters, values, p0=[sum(values)/0.18, 0.18, min(values)], sigma=values_err, absolute_sigma=False)
                params = unc.correlated_values(params, cov)
                xx = np.arange(*xlim, 0.1)
                y_fit_u = exp_fall_unc(xx, *params)
                y_fit_err = np.array([y.s for y in y_fit_u])
                y_fit = np.array([y.n for y in y_fit_u])
                paramlabel=r"$\mathrm{f}(x): "+f"{params[0]}"+r"\mathrm{e}^{"+f"{-1*params[1]}"+"x} "+f"{'+' if params[2].n > 0 else '-'}{params[2]}$"
                paramlabel = paramlabel.replace("+/-", r"\pm")
                ax.plot(xx, y_fit, linestyle="-", color="red", label=paramlabel)
                ax.fill_between(xx, y_fit - y_fit_err, y_fit + y_fit_err, alpha=0.3, color="grey")#, step="post")

                plot_tools.addLegend(ax, 1)

                outfile=f"plot_sideband_{n}_charge{idx_charge}_eta{idx_eta}"
                if args.postfix:
                    outfile += f"_{args.postfix}"
                plot_tools.save_pdf_and_png(outdir, outfile)

# extended ABCD diagnostics
def plot_chi2_extnededABCD_frf(syst_variations=False, auxiliary_info=True,  polynomial="bernstein"):
    # smoothing 1D
    info=dict(interpolate_x=False, integrate_shapecorrection_x=True, smooth_shapecorrection=True, smooth_fakerate=True, polynomial=polynomial)

    hSel_pol0 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=0, smoothing_order_shapecorrection=0)
    hSel_pol1 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=1, smoothing_order_shapecorrection=1)
    hSel_pol2 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=2, smoothing_order_shapecorrection=2)
    hSel_pol3 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=3, smoothing_order_shapecorrection=3)

    y_frf0, y_frf0_var, params_frf0, cov_frf0, chi2_frf0, ndf_frf0 = hSel_pol0.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_frf1, y_frf1_var, params_frf1, cov_frf1, chi2_frf1, ndf_frf1 = hSel_pol1.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_frf2, y_frf2_var, params_frf2, cov_frf2, chi2_frf2, ndf_frf2 = hSel_pol2.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_frf3, y_frf3_var, params_frf3, cov_frf3, chi2_frf3, ndf_frf3 = hSel_pol3.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)

    plot_chi2(chi2_frf0.ravel(), ndf_frf0, outdirFRF, xlim=(0,35), suffix=proc, outfile="chi2_pol0")
    plot_chi2(chi2_frf1.ravel(), ndf_frf1, outdirFRF, xlim=(0,35), suffix=proc, outfile="chi2_pol1")
    plot_chi2(chi2_frf2.ravel(), ndf_frf2, outdirFRF, xlim=(0,35), suffix=proc, outfile="chi2_pol2")
    plot_chi2(chi2_frf3.ravel(), ndf_frf3, outdirFRF, xlim=(0,35), suffix=proc, outfile="chi2_pol3")

def plot_chi2_extnededABCD_scf(syst_variations=False, auxiliary_info=True,  polynomial="bernstein"):
    info=dict(interpolate_x=False, integrate_shapecorrection_x=True, smooth_shapecorrection=True, smooth_fakerate=True, polynomial=polynomial)

    hSel_pol0 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=0, smoothing_order_shapecorrection=0)
    hSel_pol1 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=1, smoothing_order_shapecorrection=1)
    hSel_pol2 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=2, smoothing_order_shapecorrection=2)
    hSel_pol3 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=3, smoothing_order_shapecorrection=3)

    y_scf0, y_scf0_var, params_scf0, cov_scf0, chi2_scf0, ndf_scf0 = hSel_pol0.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_scf1, y_scf1_var, params_scf1, cov_scf1, chi2_scf1, ndf_scf1 = hSel_pol1.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_scf2, y_scf2_var, params_scf2, cov_scf2, chi2_scf2, ndf_scf2 = hSel_pol2.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_scf3, y_scf3_var, params_scf3, cov_scf3, chi2_scf3, ndf_scf3 = hSel_pol3.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)        

    plot_chi2(chi2_scf0.ravel(), ndf_scf0, outdirSCF, xlim=(0,35), suffix=proc, outfile="chi2_pol0")
    plot_chi2(chi2_scf1.ravel(), ndf_scf1, outdirSCF, xlim=(0,35), suffix=proc, outfile="chi2_pol1")
    plot_chi2(chi2_scf2.ravel(), ndf_scf2, outdirSCF, xlim=(0,35), suffix=proc, outfile="chi2_pol2")
    plot_chi2(chi2_scf3.ravel(), ndf_scf3, outdirSCF, xlim=(0,35), suffix=proc, outfile="chi2_pol3")

def plot_diagnostics_extnededABCD(syst_variations=False, auxiliary_info=True,  polynomial="bernstein"):
    # binned fakerate / shape correction
    hSel_binned = sel.FakeSelector2DExtendedABCD(h, 
        interpolate_x=False, smooth_shapecorrection=False, smooth_fakerate=False, polynomial=polynomial)#, rebin_smoothing_axis=None)
    y_frf_binned, y_frf_binned_var = hSel_binned.compute_fakeratefactor(h)
    y_scf_binned, y_scf_binned_var = hSel_binned.compute_shapecorrection(h)

    hSel_pol0 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=0, smoothing_order_shapecorrection=0)
    hSel_pol1 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=1, smoothing_order_shapecorrection=1)
    hSel_pol2 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=2, smoothing_order_shapecorrection=2)
    hSel_pol3 = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=3, smoothing_order_shapecorrection=3)

    y_frf0, y_frf0_var, params_frf0, cov_frf0, chi2_frf0, ndf_frf0 = hSel_pol0.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_frf1, y_frf1_var, params_frf1, cov_frf1, chi2_frf1, ndf_frf1 = hSel_pol1.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_frf2, y_frf2_var, params_frf2, cov_frf2, chi2_frf2, ndf_frf2 = hSel_pol2.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_frf3, y_frf3_var, params_frf3, cov_frf3, chi2_frf3, ndf_frf3 = hSel_pol3.compute_fakeratefactor(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)

    y_scf0, y_scf0_var, params_scf0, cov_scf0, chi2_scf0, ndf_scf0 = hSel_pol0.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_scf1, y_scf1_var, params_scf1, cov_scf1, chi2_scf1, ndf_scf1 = hSel_pol1.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_scf2, y_scf2_var, params_scf2, cov_scf2, chi2_scf2, ndf_scf2 = hSel_pol2.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
    y_scf3, y_scf3_var, params_scf3, cov_scf3, chi2_scf3, ndf_scf3 = hSel_pol3.compute_shapecorrection(h, syst_variations=syst_variations, auxiliary_info=auxiliary_info)        

    # some settings
    idx_ax_charge = h.axes.name.index("charge")
    idx_ax_eta = h.axes.name.index("eta")
    idx_ax_pt = h.axes.name.index("pt")
    x_edges = hSel_binned.h_shapecorrection.axes["pt"].edges
    x_widths = np.diff(x_edges)/2
    x_centers = x_edges[:-1]+x_widths

    mt_edges = hSel_binned.h_shapecorrection.axes["mt"].edges
    mt_widths = np.diff(mt_edges)/2
    mt_centers = mt_edges[:-1]+mt_widths
    if h.axes["mt"].traits.overflow:
        mt_centers = np.append(mt_centers, mt_centers[-1]+np.diff(mt_centers[-2:]))
        mt_widths = np.append(mt_widths, np.diff(mt_centers[-2:])/2)
        mt_edges = np.append(mt_edges, mt_edges[-1]+np.diff(mt_edges[-2:]))

    if polynomial=="power":
        mtlim = [min(mt_centers-mt_widths), max(mt_centers+mt_widths)]
        xlim = [min(x_centers-x_widths), max(x_centers+x_widths)]
    elif polynomial=="bernstein":
        x_widths = x_widths/(x_edges[-1]-x_edges[0])
        x_centers = (x_centers-x_edges[0])/(x_edges[-1]-x_edges[0])
        mtlim = [0, 1]
        xlim = [0, 1]

    mt = np.linspace(*mtlim, 101)
    xx = np.linspace(*xlim, 101)

    h_pt = h.project("pt")
    logy=False
    etavar="|\eta|"

    # # smoothed shapecorrection in 2D, try all possible combinations
    info=dict(interpolate_x=True, integrate_shapecorrection_x=False, smooth_shapecorrection=True, smooth_fakerate=True, polynomial=polynomial)
    ylabel = styles.xlabels["mt"]
    xlabel = styles.xlabels["pt"]
    for i in range(3):
        print(f"i = {i}")
        for j in itertools.product([0,1,2], repeat=i+1):
            print(f"j = {j}")

            hSel_pol = sel.FakeSelector2DExtendedABCD(h, **info, smoothing_order_fakerate=0, interpolation_order=i, smoothing_order_shapecorrection=list(j))
            y_scf, y_scf_var, ps, cs, chi2s, ndf = hSel_pol.compute_shapecorrection(h, syst_variations=False, auxiliary_info=True)
            plot_chi2(chi2_scf.ravel(), ndf_scf, outdirSCF, xlim=(0,150), suffix=proc, outfile=f"chi2_mtPol{i}_ptPol{'_'.join([str(ij) for ij in j])}")

            for idx_charge, charge_bins in enumerate(h.axes["charge"]):
                for idx_eta, eta_bins in enumerate(h.axes["eta"]):
                    outdirSCFBin = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/shapecorrection_factor//charge{idx_charge}_eta{idx_eta}")

                    if i==0 and all(j==0):
                        # plot the binned once
                        slices=[slice(None),]*len(y_scf_binned.shape)
                        slices[idx_ax_charge] = idx_charge
                        slices[idx_ax_eta] = idx_eta
                        y = y_scf_binned[*slices]
                        fig = plot_tools.makePlot2D(values=y, xlabel=xlabel, ylabel=ylabel, xedges=x_edges, yedges=mt_edges, cms_label=args.cmsDecor, zlim=[0.5,1.5])
                        outfile = f"scf2D_binned"
                        if args.postfix:
                            outfile += f"_{args.postfix}"
                        plot_tools.save_pdf_and_png(outdirSCFBin, outfile)


                    chi2 = chi2s[idx_eta,idx_charge]

                    p = ps[idx_eta,idx_charge,:]
                    c = cs[idx_eta,idx_charge,:,:]
                    
                    y_fit = hSel_pol.f_scf(mt, xx, p)

                    fig = plot_tools.makePlot2D(values=y_fit, xlabel=xlabel, ylabel=ylabel, xedges=xx, yedges=mt, cms_label=args.cmsDecor, zlim=[0.5,1.5], 
                        plot_title="$\chi2/"+f"{ndf} = "+str(round(chi2))+"$")

                    outfile=f"scf2D_mtPol{i}_ptPol{'_'.join([str(ij) for ij in j])}"
                    if args.postfix:
                        outfile += f"_{args.postfix}"
                    plot_tools.save_pdf_and_png(outdirSCFBin, outfile)

                    continue # just do first bin
                continue # just do first bin

    # smoothed fakerate factors in 1D
    for idx_charge, charge_bins in enumerate(h.axes["charge"]):
        for idx_eta, eta_bins in enumerate(h.axes["eta"]):
            slices=[slice(None),]*len(y_frf_binned.shape)
            slices[idx_ax_charge] = idx_charge
            slices[idx_ax_eta] = idx_eta

            linestyles = ["-", "-", "--", "--", ":"]
            colors = mpl.colormaps["tab10"]
            outfile = f"charge{idx_charge}_eta{idx_eta}"
            binlabel=f"${round(eta_bins[0],1)} < {etavar} < {round(eta_bins[1],1)}$"

            # fakerate factors
            y = y_frf_binned[*slices]
            yerr = y_frf_binned_var[*slices]**0.5

            if logy:
                ylim = [max(0.1, min(y-yerr)*0.5), min(5,max(y+yerr))*2]
            else:
                # ylim = [max(0, min(y-yerr)*0.9), min(5,max(y+yerr))*1.1]
                ylim = [0,5]
            
            fig, ax1 = plot_tools.figure(h_pt, ylabel="Fakerate factor", xlabel="$p_\mathrm{T}$ (GeV)", cms_label=args.cmsDecor, xlim=xlim, ylim=ylim, logy=logy)

            ax1.errorbar(x_centers, y, xerr=x_widths, yerr=yerr, marker="", linestyle='none', color="k", label=binlabel)

            for i, (ps, cs, f, chi2, ndf) in enumerate((
                # (params_frf0, cov_frf0, hSel_pol0.f_frf, chi2_frf0, ndf_frf0),
                # (params_frf1, cov_frf1, hSel_pol1.f_frf, chi2_frf1, ndf_frf1),
                # (params_frf2, cov_frf2, hSel_pol2.f_frf, chi2_frf2, ndf_frf2),
                (params_frf3, cov_frf3, hSel_pol3.f_frf, chi2_frf3, ndf_frf3),
            )):
                chi2 = chi2[idx_eta,idx_charge]

                p = ps[idx_eta,idx_charge,:]
                c = cs[idx_eta,idx_charge,:,:]
                pu = unc.correlated_values(p, c)
                y_fit_u = f(xx, pu)
                y_fit_err = np.array([y.s for y in y_fit_u])
                y_fit = np.array([y.n for y in y_fit_u])
                # paramlabel = r"$\mathrm{f}(x)= "+f"{p[0]}"
                # if len(p)>1:
                #     paramlabel += f"{'+' if p[1].n > 0 else ''}{p[1]}x"
                # if len(p)>2:
                #     paramlabel += f"{'+' if p[2].n > 0 else ''}{p[2]}x^2"
                # if len(p)>3:
                #     paramlabel += f"{'+' if p[3].n > 0 else ''}{p[3]}x^3"
                paramlabel = r"$\chi^2/\mathrm{ndf} = "+f"{round(chi2,1)}/{ndf} $"
                paramlabel = paramlabel.replace("+/-", r"\pm")

                paramlabel = f"$n = {[round(x,1) for x in p]}$"

                ax1.plot(xx, y_fit, linestyle="-", label=paramlabel, color="black")#, color=colors(i))
                ax1.fill_between(xx, y_fit - y_fit_err, y_fit + y_fit_err, alpha=0.3, color="black")#colors(i))

                # plot eigenvector variations
                force_positive=True
                p_vars_up = sel.get_eigen_variations(p, c, sign=1, force_positive=force_positive)
                p_vars_down = sel.get_eigen_variations(p, c, sign=-1, force_positive=force_positive)
                # p_vars =ps_vars[idx_eta,idx_charge,:,:]
                for j, (p_up, p_down) in enumerate(zip(p_vars_up, p_vars_down)):
                    y_fit_up = f(xx, p_up)
                    y_fit_down = f(xx, p_down)
                    ax1.plot(xx, y_fit_up, linestyle="--", color=colors(i), label=f"$e^{{up}}_{j} = {[round(x,1) for x in p_up]}$")
                    ax1.plot(xx, y_fit_down, linestyle=":", color=colors(i+1), label=f"$e^{{dn}}_{j} = {[round(x,1) for x in p_down]}$")

            ax1.text(1.0, 1.003, styles.process_labels[proc], transform=ax1.transAxes, fontsize=30,
                    verticalalignment='bottom', horizontalalignment="right")
            plot_tools.addLegend(ax1, ncols=2, text_size=20, loc="upper left")
            plot_tools.fix_axes(ax1, ax2=None)

            if args.postfix:
                outfile += f"_{args.postfix}"

            plot_tools.save_pdf_and_png(outdirFRF, outfile)

            # plots for shapecorrection factors
            # 1D plots in mt
            for idx_pt, pt_bins in enumerate(h.axes["pt"]):
                slices[idx_ax_pt] = idx_pt

                linestyles = ["-", "-", "--", "--", ":"]
                colors = mpl.colormaps["tab10"]
                outfile = f"charge{idx_charge}_eta{idx_eta}_pt{idx_pt}"
                binlabel=f"${round(eta_bins[0],1)} < {etavar} < {round(eta_bins[1],1)}; {round(pt_bins[0])} < "+r"p_\mathrm{T}"+f" < {round(pt_bins[1])}$"

                # shape correction factors
                y = y_scf[*slices]
                yerr = y_scf_var[*slices]**0.5

                if logy:
                    ylim = [0.1, max(y+yerr)*2]
                else:
                    ylim = [0, max(y+yerr)*1.1]
                
                fig, ax1 = plot_tools.figure(h_pt, ylabel="Shape correction factor", xlabel="$m_\mathrm{T}$ (GeV)", cms_label=args.cmsDecor, xlim=mtlim, ylim=ylim, logy=logy)

                ax1.errorbar(mt_centers, y, xerr=mt_widths, yerr=yerr, marker="", linestyle='none', color="k", label=binlabel)

                for i, (ps, cs, f, chi2, ndf) in enumerate((
                    (params_scf0, cov_scf0, f_scf0, chi20, ndf0),
                    (params_scf1, cov_scf1, f_scf1, chi21, ndf1),
                    (params_scf2, cov_scf2, f_scf2, chi22, ndf2),
                    (params_scf3, cov_scf3, f_scf3, chi23, ndf3),
                )):
                    chi2 = chi2[idx_eta,idx_pt,idx_charge]

                    p = ps[idx_eta,idx_pt,idx_charge,:]
                    c = cs[idx_eta,idx_pt,idx_charge,:,:]
                    p = unc.correlated_values(p, c)
                    y_fit_u = f(mt, p)
                    y_fit_err = np.array([y.s for y in y_fit_u])
                    y_fit = np.array([y.n for y in y_fit_u])
                    # paramlabel = r"$\mathrm{f}(x)= "+f"{p[0]}"
                    # if len(p)>1:
                    #     paramlabel += f"{'+' if p[1].n > 0 else ''}{p[1]}x"
                    # if len(p)>2:
                    #     paramlabel += f"{'+' if p[2].n > 0 else ''}{p[2]}x^2"
                    # if len(p)>3:
                    #     paramlabel += f"{'+' if p[3].n > 0 else ''}{p[3]}x^3"
                    paramlabel = r"$\chi^2/\mathrm{ndf} = "+f"{round(chi2,1)}/{ndf} $"
                    paramlabel = paramlabel.replace("+/-", r"\pm")

                    ax1.plot(mt, y_fit, linestyle="-", color=colors(i), label=paramlabel)
                    ax1.fill_between(mt, y_fit - y_fit_err, y_fit + y_fit_err, alpha=0.3, color=colors(i))

                ax1.text(1.0, 1.003, styles.process_labels[proc], transform=ax1.transAxes, fontsize=30,
                        verticalalignment='bottom', horizontalalignment="right")
                plot_tools.addLegend(ax1, ncols=2, text_size=20)
                plot_tools.fix_axes(ax1, ax2=None)

                if args.postfix:
                    outfile += f"_{args.postfix}"

                plot_tools.save_pdf_and_png(outdirSCF, outfile)


### plot closure
def plot_closure(h, outdir, suffix="", outfile=f"closureABCD", ratio=True, proc="", ylabel="a.u.", smoothed=False):
    fakerate_integration_axes = [a for a in ["eta","pt","charge"] if a not in args.vars]
    threshold = args.xBinsSideband[-1]
    
    h = hh.rebinHist(h, "pt", [26, 31, 40, 56])

    hss=[]
    labels=[]

    info=dict(rebin_smoothing_axis=None)

    # signal selection
    hSel_sig = sel.SignalSelectorABCD(h, **info)
    hD_sig = hSel_sig.get_hist(h)
    hss.append(hD_sig)
    labels.append("D")
    
    # simple ABCD
    hSel_simple = sel.FakeSelectorSimpleABCD(h, **info)
    hD_simple = hSel_simple.get_hist(h)
    hss.append(hD_simple)
    labels.append("simple")

    # # interpolated ABCD
    # # interpolate in x
    # infoX=dict(fakerate_integration_axes=fakerate_integration_axes, integrateHigh=True, 
    #     smoothing_axis_name=args.smoothingAxisName, smoothing_order=args.smoothingOrder,
    # )
    # hD_Xpol1 = sel.fakeHistExtendedABCD(h, **info, order=1, **infoX)
    # hss.append(hD_Xpol1)
    # labels.append("pol1(x)")
    # # hD_pol2 = sel.fakeHistExtendedABCD(h, **info, order=2, **infoX)
    # 
    # # interpolate in y
    # infoY=dict(fakerate_integration_axes=fakerate_integration_axes, integrateHigh=True, 
    #     smoothing_axis_name=args.smoothingAxisName, smoothing_order=args.smoothingOrder,
    # )
    # hD_Ypol1 = sel.fakeHistExtendedABCD(h, **info, order=1, **infoY)
    # hss.append(hD_Ypol1)
    # labels.append("pol1(y)")

    # extended ABCD in 5 control regions
    if smoothed:
        hSel_ext5 = sel.FakeSelector1DExtendedABCD(h, **info)
        hD_ext5 = hSel_ext5.get_hist(h)
        hss.append(hD_ext5)
        labels.append("ext(5) smoothed")
    else:
        hSel_ext5 = sel.FakeSelector1DExtendedABCD(h, **info, smooth_fakerate=False, upper_bound_y=None)
        hD_ext5 = hSel_ext5.get_hist(h)
        hss.append(hD_ext5)
        labels.append("ext(5) binned")

        # hSel_ext5 = sel.FakeSelector1DExtendedABCD(h, **info, smooth_fakerate=False upper_bound_y=hist.overflow)
        # hD_ext5 = hSel_ext5.get_hist(h)
        # hss.append(hD_ext5)
        # labels.append("ext(5) binned (iso<0.45)")

    # extended ABCD in 8 control regions
    if smoothed:
        hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info)
        hD_ext8 = hSel_ext8.get_hist(h)
        hss.append(hD_ext8)
        labels.append("ext(8) smoothed")
    else:
        hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, upper_bound_y=None,
            integrate_shapecorrection_x=False, interpolate_x=False, smooth_shapecorrection=False, smooth_fakerate=False)
        hD_ext8 = hSel_ext8.get_hist(h)
        hss.append(hD_ext8)
        labels.append("ext(8) binned")

        # hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, upper_bound_y=hist.overflow,
        #     integrate_shapecorrection_x=False, interpolate_x=False, smooth_shapecorrection=False, smooth_fakerate=False)
        # hD_ext8 = hSel_ext8.get_hist(h)
        # hss.append(hD_ext8)
        # labels.append("ext(8) binned (iso<0.45)")

        # hSel_ext8 = sel.FakeSelector2DExtendedABCD(h, **info, integrate_shapecorrection_x=True, interpolate_x=False, smooth_shapecorrection=False, smooth_fakerate=False)
        # hD_ext8 = hSel_ext8.get_hist(h)
        # hss.append(hD_ext8)
        # labels.append("ext(8) binned (mT integrated)")

    linestyles = ["-", "-", "--", "--", ":"]
    colors = mpl.colormaps["tab10"]
    
    if "charge" in hss[0].axes.name and len(hss[0].axes["charge"])==1:
        hss = [h[{"charge":slice(None,None,hist.sum)}] for h in hss]

    axes = hss[0].axes.name

    if len(axes)>1:
        hss = [hh.unrolledHist(h, obs=axes[::-1]) for h in hss]

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
                                             rlabel=f"1/{labels[0]}", rrange=args.rrange, 
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
    parser.add_argument("--rebinBeforeSelection", action='store_true', help="Rebin before the selection operation (e.g. before fake rate computation), default if after")
    parser.add_argument("--rrange", type=float, nargs=2, default=[0.25, 1.75], help="y range for ratio plot")
    # x-axis for ABCD method
    parser.add_argument("--xAxisName", type=str, help="Name of x-axis for ABCD method", default="mt")
    parser.add_argument("--xBinsSideband", type=float, nargs='*', help="Binning of x-axis for ABCD method in sideband region", default=[0,2,5,9,14,20,27,40])
    parser.add_argument("--xBinsSignal", type=float, nargs='*', help="Binning of x-axis of ABCD method in signal region", default=[50,70,120])
    parser.add_argument("--xOrder", type=int, default=2, help="Order in x-axis for fakerate parameterization")
    # y-axis for ABCD method
    parser.add_argument("--yAxisName", type=str, help="Name of y-axis for ABCD method", default="iso")
    parser.add_argument("--yBinsSideband", type=float, nargs='*', help="Binning of y-axis of ABCD method in sideband region", default=[0.15,0.2,0.25,0.3])
    parser.add_argument("--yBinsSignal", type=float, nargs='*', help="Binning of y-axis of ABCD method in signal region", default=[0,0.15])
    parser.add_argument("--yOrder", type=int, default=2, help="Order in y-axis for fakerate parameterization")
    # axis for smoothing
    parser.add_argument("--smoothingAxisName", type=str, help="Name of second axis for ABCD method, 'None' means 1D fakerates", default=None)
    parser.add_argument("--smoothingOrder", type=int, nargs='*', default=[2,1,0], help="Order in second axis for fakerate parameterization")

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)

    groups = Datagroups(args.infile, excludeGroups=None, extendedABCD=True, integrateHigh=True)

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
            plot_closure(h, outdir, suffix=f"{proc}", proc=proc, smoothed=False)
            plot_closure(h, outdir, suffix=f"{proc}_smoothed", proc=proc, smoothed=True)

        # plot fakerate factors for full extended ABCD method
        outdirFRF = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/fakerate_factor/")
        outdirSCF = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/shapecorrection_factor/")
        plot_diagnostics_extnededABCD(outdirFRF)

        # plot 1D shape in signal and different sideband regions
        outdirDistributions = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/distributions_sideband/")
        plot_abcd_fits(h, outdirDistributions)

        # plot chi2 of leastsquare
        hLowMT = hh.rebinHist(h, args.xAxisName, args.xBinsLow, args.rebinBeforeSelection)
        params, cov, frf, chi2, ndf = sel.compute_fakerate(hLowMT, axis_name=args.xAxisName, overflow=True, use_weights=True, 
            order=args.xOrder, axis_name_2=args.xsmoothingAxisName, order_2=args.xsmoothingOrder, auxiliary_info=True)
        plot_chi2(chi2.ravel(), ndf, outdir, xlim=(0,10), suffix=proc)

        # plot interpolation parameters
        outdir1D = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_1D_{proc}/")
        outdir2D = output_tools.make_plot_dir(args.outpath, f"{args.outfolder}/plots_2D_{proc}/")
        for idx_charge, charge_bins in enumerate(hLowMT.axes["charge"]):
            logger.info(f"Make parameter validation plots for charge index {idx_charge}")

            sign = "" if len(hLowMT.axes["charge"])==1 else "-" if idx_charge==0 else "+" 
            
            if args.xsmoothingAxisName:
                idx_p=0
                outdir_p = output_tools.make_plot_dir(outdir, f"params")
                for ip, (pName, pRange) in enumerate([("offset", None), ("slope", (-0.02, 0.02)), ("quad", (-0.0004, 0.0004))]):
                    if ip > args.xOrder:
                        break
                    for ip2, (pName2, pRange2) in enumerate([("offset", None), ("slope", (-0.02, 0.02)), ("quad", (-0.0004, 0.0004))]):
                        if ip2 > args.xsmoothingOrder[ip]:
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


