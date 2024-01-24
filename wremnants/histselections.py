import hist
import numpy as np
from utilities.io_tools.input_tools import safeOpenRootFile, safeGetRootObject
from utilities import boostHistHelpers as hh
from utilities import common, logging
import narf
import ROOT

from scipy.optimize import curve_fit

logger = logging.child_logger(__name__)

hist_map = {
    "eta_pt" : "nominal",
    "eta" : "nominal",
    "pt" : "nominal",
    "mll" : "nominal",
    "ptll" : "nominal",
    "ptll_mll" : "nominal",
}

def fakeHistABCD(h, thresholdMT=40.0, fakerate_integration_axes=[], axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
    # integrateMT=False keeps the mT axis in the returned histogram (can be used to have fakes vs mT)

    nameMT, failMT, passMT = get_mt_selection(h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT)

    if any(a in h.axes.name for a in fakerate_integration_axes if a not in [common.passIsoName, nameMT]):
        fakerate_axes = [n for n in h.axes.name if n not in [*fakerate_integration_axes, common.passIsoName, nameMT]]
        logger.debug(f"Integrate over fakerate integration axes {fakerate_integration_axes} and keep {fakerate_axes}")
        hPassIsoFailMT = h[{**common.passIso, nameMT: failMT}].project(*fakerate_axes)
        hFailIsoFailMT = h[{**common.failIso, nameMT: failMT}].project(*fakerate_axes)
    else:
        hPassIsoFailMT = h[{**common.passIso, nameMT: failMT}]
        hFailIsoFailMT = h[{**common.failIso, nameMT: failMT}]

    hFRF = hh.divideHists(hPassIsoFailMT, hFailIsoFailMT, cutoff=1, createNew=True)   

    return hh.multiplyHists(hFRF, h[{**common.failIso, nameMT: passMT}])

def fakeHistSimultaneousABCD(h, thresholdMT=40.0, fakerate_integration_axes=[], axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
    if h.storage_type == hist.storage.Weight:
        # setting errors to 0
        h.view(flow=True)[...] = np.stack((h.values(flow=True), np.zeros_like(h.values(flow=True))), axis=-1)

    nameMT, failMT, passMT = get_mt_selection(h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT)

    if common.passIsoName not in h.axes.name or nameMT not in h.axes.name:
        raise RuntimeError(f'{common.passIsoName} and {nameMT} expected to be found in histogram, but only have axes {h.axes.name}')

    # axes in the correct ordering
    axes = [ax for ax in h.axes.name if ax not in [nameMT, common.passIsoName]]
    axes += [common.passIsoName, nameMT]

    if set(h.axes.name) != set(axes):
        logger.warning(f"Axes in histogram '{h.axes.name}' are not the same as required '{axes}' or in a different order than expected, try to project")
        h = h.project(*axes)

    # set the expected values in the signal region
    slices = [passMT if n==nameMT else 1 if n==common.passIsoName else slice(None) for n in h.axes.name]
    h.values(flow=True)[*slices] = fakeHistABCD(h, thresholdMT, fakerate_integration_axes, axis_name_mt, integrateLowMT, integrateHighMT).values(flow=True)

    return h

def fakeHistIsoRegion(h, scale=1.):
    #return h[{"iso" : 0.3j, "mt" : hist.rebin(10)}]*scale
    return h[{"iso" : 4}]*scale

def fakeHistIsoRegionIntGen(h, scale=1.):
    if not "qTgen" in [ax.name for ax in h.axes]:
        return h[{"iso" : 4}]*scale
    s = hist.tag.Slicer()
    return h[{"iso" : 0, "qTgen" : s[::hist.sum]}]

def signalHistWmass(h, thresholdMT=40.0, charge=None, passIso=True, passMT=True, axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False, genBin=None):
    if genBin != None:
        h = h[{"recoil_gen" : genBin}]

    nameMT, fMT, pMT = get_mt_selection(h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT)

    sel = {common.passIsoName:passIso, nameMT: pMT if passMT else fMT}
    if charge in [-1, 1]:
        sel.update({"charge" : -1j if charge < 0 else 1j})

    # remove ax slice if the ax does not exist
    for key in sel.copy().keys():
        if not key in [ax.name for ax in h.axes]: 
            del sel[key]

    return h[sel]

# the following are utility wrapper functions for signalHistWmass with proper region selection
def histWmass_failMT_passIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, True, False)

def histWmass_failMT_failIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, False, False)

def histWmass_passMT_failIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, False, True)

def histWmass_passMT_passIso(h, thresholdMT=40.0, charge=None):
    return signalHistWmass(h, thresholdMT, charge, True, True)

# TODO: Not all hists are made with these axes
def signalHistLowPileupW(h):
    if not "qTgen" in [ax.name for ax in h.axes]:
        return h[{"iso" : 0}]
    s = hist.tag.Slicer()
    return h[{"iso" : 0, "qTgen" : s[::hist.sum]}]
    
def signalHistLowPileupZ(h):
    return h

def get_mt_selection(h, thresholdMT=40.0, axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
    if axis_name_mt in h.axes.name:
        s = hist.tag.Slicer()
        high = h.axes[axis_name_mt].index(thresholdMT)
        failMT = s[:high:hist.sum] if integrateLowMT else s[:high:]
        passMT = s[high:hist.sum] if integrateHighMT else s[high:]
        nameMT = axis_name_mt
    else:
        failMT = 0
        passMT = 1
        nameMT = common.passMTName

    return nameMT, failMT, passMT

def unrolledHist(h, obs=["pt", "eta"], binwnorm=None):
    if obs is not None:
        hproj = h.project(*obs)
    else:
        hproj = h

    if binwnorm:        
        binwidths = np.outer(*[np.diff(e.squeeze()) for e in hproj.axes.edges]).flatten()
        scale = binwnorm/binwidths
    else:
        scale = 1

    bins = np.product(hproj.axes.size)
    newh = hist.Hist(hist.axis.Integer(0, bins), storage=hproj._storage_type())
    newh[...] = np.ravel(hproj)*scale
    return newh

def applyCorrection(h, scale=1.0, offsetCorr=0.0, corrFile=None, corrHist=None, createNew=False):
    # originally intended to apply a correction differential in eta-pt
    # corrHist is a TH3 with eta-pt-charge
    # scale is just to apply an additional constant scaling (or only that if ever needed) before the correction from corrHist
    # offsetCorr is for utility, to add to the correction from the file, e.g. if the histogram is a scaling of x (e.g. 5%) one has to multiply the input histogram h by 1+x, so offset should be 1
    boost_corr = None
    if corrFile and corrHist:
        ## TDirectory.TContext() should restore the ROOT current directory to whatever it was before a new ROOT file was opened
        ## but it doesn't work at the moment, apparently the class miss the __enter__ member and the usage with "with" fails
        #with ROOT.TDirectory.TContext():
        f = safeOpenRootFile(corrFile, mode="READ")
        corr = safeGetRootObject(f, corrHist, detach=True)
        if offsetCorr:
            offsetHist = corr.Clone("offsetHist")
            ROOT.wrem.initializeRootHistogram(offsetHist, offsetCorr)
            corr.Add(offsetHist)
        f.Close()
        boost_corr = narf.root_to_hist(corr)
    # note: in fact hh.multiplyHists already creates a new histogram
    hnew = hh.scale(h, scale, createNew)
    if boost_corr:
        hnew = hh.multiplyHists(hnew, boost_corr)

    return hnew

import uncertainties as unc
from uncertainties import unumpy as unp

def exp_fall_unc(x, a, b, c):
    return a * unp.exp(-b * x) + c

def exp_fall(x, a, b, c):
    return a * np.exp(-b * x) + c

def fit_multijet_bkg(h):
    logger.info(f"Fit exp falling function --- ")
    x = h.axes["pt"].centers
    y = h.values()
    y_err = np.sqrt(h.variances())

    p0=[sum(y)/0.15, 0.15, min(y)] # initial guesses
    if sum(y==0)>1:
        # do a first fit with only keeping the points that are not empty , to get an initial estimate on the parameters
        x_pruned = x[y!=0]
        y_err_pruned = y_err[y!=0]
        y_pruned = y[y!=0]
        params, cov = curve_fit(exp_fall, x_pruned, y_pruned, p0=p0, sigma=y_err_pruned, absolute_sigma=False)
        p0=params
        logger.info(f"Initial fit params={params}")

    params, cov = curve_fit(exp_fall, x, y, p0=p0, sigma=y_err, absolute_sigma=False)

    params = unc.correlated_values(params, cov)
    y_fit_u = exp_fall_unc(x, *params)
    y_fit_err = np.array([y.s for y in y_fit_u])
    y_fit = np.array([y.n for y in y_fit_u])
    
    chisq = sum(((y - y_fit) / y_fit_err) ** 2) # use statistical error as expected from fit to avoid division through 0
    ndf = len(x)-len(params)
    logger.info(f"Fit result with chi2/ndf = {chisq:.2f}/{ndf} = {(chisq/ndf):.2f}")
    logger.info(f"Fit params={params}")

    return y_fit_u

def get_multijet_bkg_closure(h, fake_axes, slices=[slice(None)], variances=True):

    nameMT, failMT, passMT = get_mt_selection(h)

    hA = h[{**common.failIso, nameMT: failMT}].project(*fake_axes)[*slices]
    hB = h[{**common.passIso, nameMT: failMT}].project(*fake_axes)[*slices]
    hC = h[{**common.failIso, nameMT: passMT}].project(*fake_axes)[*slices]
    hD = h[{**common.passIso, nameMT: passMT}].project(*fake_axes)[*slices]

    yA_fit = fit_multijet_bkg(hA)
    yB_fit = fit_multijet_bkg(hB)
    yC_fit = fit_multijet_bkg(hC)
    yD_fit = fit_multijet_bkg(hD)

    yClosure = yD_fit / (yC_fit*yB_fit/yA_fit)
    if variances:
        return np.array([(v.n,v.s**2) for v in yClosure])
    else:
        return np.array([v.n for v in yClosure])
