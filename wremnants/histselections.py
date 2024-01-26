import hist
import numpy as np
from utilities.io_tools.input_tools import safeOpenRootFile, safeGetRootObject
from utilities import boostHistHelpers as hh
from utilities import common, logging
import narf
import ROOT

import pdb

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

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [n for n in h.axes.name if n not in [*fakerate_integration_axes, common.passIsoName, nameMT]]
        hPassIsoFailMT = h[{**common.passIso, nameMT: failMT}].project(*fakerate_axes)
        hFailIsoFailMT = h[{**common.failIso, nameMT: failMT}].project(*fakerate_axes)
    else:
        hPassIsoFailMT = h[{**common.passIso, nameMT: failMT}]
        hFailIsoFailMT = h[{**common.failIso, nameMT: failMT}]

    hFRF = hh.divideHists(hPassIsoFailMT, hFailIsoFailMT, cutoff=1, createNew=True)   

    return hh.multiplyHists(hFRF, h[{**common.failIso, nameMT: passMT}])

def fakeHistExtendedABCD(h, thresholdMT=40.0, fakerate_integration_axes=[], axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
    logger.debug("Compute fakes with extended ABCD method")
    # integrateMT=False keeps the mT axis in the returned histogram (can be used to have fakes vs mT)

    # allow for a slope in the FRF with mt
    # 1) calculate FRF(mt) in each bin of the fakerate axis
    # 2) obtain offset and slope from lowMT bins using weighted least square fit https://en.wikipedia.org/wiki/Weighted_least_squares
    # 3) apply the FRF(mt) in high mt bins

    if h.axes[axis_name_mt].traits.underflow:
        raise NotImplementedError(f"Underflow bins for axis {axis_name_mt} is not supported")

    overflow_mt = h.axes[axis_name_mt].traits.overflow

    # hLowMT = h[{axis_name_mt: slice(None,complex(0,thresholdMT), hist.rebin(20))}]
    hLowMT = hh.rebinHist(h, axis_name_mt, [0,15,thresholdMT])
    # hLowMT = hh.rebinHist(h, axis_name_mt, [0,10,20,thresholdMT])
    # hLowMT = hh.rebinHist(h, axis_name_mt, [0,4,11,21,thresholdMT])

    if any(a in h.axes.name for a in fakerate_integration_axes if a != axis_name_mt):
        logger.info(f"Project {fakerate_axes}")
        fakerate_axes = [n for n in h.axes.name if n not in [*fakerate_integration_axes, common.passIsoName]]
        hLowMT = hLowMT.project(*fakerate_axes)

    hLowMTPassIso = hLowMT[common.passIso]
    hLowMTFailIso = hLowMT[common.failIso]

    hFRF = hh.divideHists(hLowMTPassIso, hLowMTFailIso, cutoff=0.1, createNew=True, flow=False)

    y = hFRF.values(flow=True)

    idx_ax_mt = hFRF.axes.name.index(axis_name_mt)
    if idx_ax_mt != len(hFRF.axes)-1:
        y = np.moveaxis(y, idx_ax_mt, -1)
    if overflow_mt:
        # stupit, but if the mt has overflow even if lowMT range is selected we get back an overflow bin with zeros (even when we use the hist slicing methods), strip them off
        y = y[...,:-1]

    x = hFRF.axes[axis_name_mt].centers
    x = np.broadcast_to(x, y.shape)
    x0 = np.ones(x.shape)
    if h.storage_type == hist.storage.Weight:
        # transform with weights
        w = 1./np.sqrt(hFRF.variances(flow=True))
        if idx_ax_mt != len(hFRF.axes)-1:
            w = np.moveaxis(w, idx_ax_mt, -1)
        if overflow_mt:
            w = w[...,:-1]
        x0 = x0*w
        x = x*w
        y = y*w
    else:
        logger.warning("Using extendedABCD on histogram without uncertainties, make an unweighted linear squared solution.")

    XTY = np.stack(((x0*y).sum(axis=-1), (x*y).sum(axis=-1)), axis=-1)

    # parameter matrix X with the "parameter" axis (here 2 parameters for offset and slope)
    X = np.stack((x0, x), axis=-1)
    # compute the transpose of X for the mt and parameter axes 
    XT = np.transpose(X, axes=(*range(len(X.shape)-2), len(X.shape)-1, len(X.shape)-2))
    XTX = XT @ X
    # compute the inverse of the matrix in each bin (reshape to make last two axes contiguous, reshape back after inversion), 
    # this term is also the covariance matrix for the parameters
    XTXinv = np.linalg.inv(XTX.reshape(-1,*XTX.shape[-2:]))
    XTXinv = XTXinv.reshape((*XT.shape[:-2],*XTXinv.shape[-2:])) 

    params = np.einsum('...ij,...j->...i', XTXinv, XTY)

    hHighMTFailIso = h[{**common.failIso, axis_name_mt: slice(complex(0,thresholdMT),None)}]

    val_c = hHighMTFailIso.values(flow=True)
    idx_ax_mt = hHighMTFailIso.axes.name.index(axis_name_mt)
    if idx_ax_mt != len(hHighMTFailIso.axes)-1:
        val_c = np.moveaxis(val_c, idx_ax_mt, -1)

    x_c = hHighMTFailIso.axes[axis_name_mt].centers
    if overflow_mt:
        # need an x point for the overflow bin, mirror distance between two last bins 
        x_c = np.append(x_c, x_c[-1]+np.diff(x_c[-2:]))

    x_c = np.broadcast_to(x_c, val_c.shape)
    FRF = params[...,1,np.newaxis] * x_c + params[...,0,np.newaxis]
    val_d = val_c * FRF

    if integrateHighMT:
        hHighMTPassIso = hist.Hist(*[a for a in hHighMTFailIso.axes if a.name != axis_name_mt], storage=h.storage_type())

        if h.storage_type == hist.storage.Weight:
            # variance from error propagation formula from parameters (first sum bins, then multiply with parameter error)
            var_d = (val_c*x_c).sum(axis=-1)**2 * XTXinv[...,1,1] + (val_c).sum(axis=-1)**2 * XTXinv[...,0,0] + 2*(val_c*x_c).sum(axis=-1)*val_c.sum(axis=-1)*XTXinv[...,1,0] 
            # variance coming from statistical uncertainty in C (indipendent bin errors, sum of squares)
            var_c = hHighMTFailIso.variances(flow=True)
            if idx_ax_mt != len(hHighMTFailIso.axes)-1:
                var_c = np.moveaxis(var_c, idx_ax_mt, -1)
            var_d += (FRF**2 * var_c).sum(axis=-1)
            val_d = np.stack((val_d.sum(axis=-1), var_d), axis=-1)
    else:
        hHighMTPassIso = hHighMTFailIso.copy()

        if idx_ax_mt != len(hHighMTFailIso.axes)-1:
            val_d = np.moveaxis(val_d, -1, idx_ax_mt) # move the axis back

        if h.storage_type == hist.storage.Weight:
            logger.warning("Using extendedABCD as function of mt without integration of mt will give fakes with incorrect bin by bin uncertainties!")
            # Wrong but easy solution for the moment, just omit the sums TODO: implement proper solution
            # variance from error propagation formula from parameters
            var_d = (val_c*x_c)**2 * XTXinv[...,1,1,np.newaxis] + (val_c)**2 * XTXinv[...,0,0,np.newaxis] + 2*(val_c*x_c)*val_c*XTXinv[...,1,0,np.newaxis] 
            # variance coming from statistical uncertainty in C (indipendent bin errors, sum of squares)
            var_c = hHighMTFailIso.variances(flow=True)
            if idx_ax_mt != len(hHighMTFailIso.axes)-1:
                var_c = np.moveaxis(var_c, idx_ax_mt, -1)
            var_d += (FRF**2 * var_c)
            if idx_ax_mt != len(hHighMTFailIso.axes)-1:
                var_d = np.moveaxis(var_d, -1, idx_ax_mt) # move the axis back
            val_d = np.stack((val_d, var_d), axis=-1)

    hHighMTPassIso.view(flow=True)[...] = val_d

    return hHighMTPassIso


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
