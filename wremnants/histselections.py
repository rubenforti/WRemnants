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

def compute_fakerate(hLow, axis_name="mt", overflow=True, use_weights=True, order=1, auxiliary_info=False,
    axis_name_2=None, order_2=None,
):
    logger.debug(f"Do the fakerate in order: {order}")
    # fakerate parameterized in one variable, mt, depending on order of polinomial: f(mt) = a + b*mt + c*mt**2 ... 
    hLowPass = hLow[common.passIso]
    hLowFail = hLow[common.failIso]
    hFRF = hh.divideHists(hLowPass, hLowFail, cutoff=0.1, createNew=True, flow=False)
    y = hFRF.values(flow=True)
    idx_ax = hFRF.axes.name.index(axis_name)
    if idx_ax != len(hFRF.axes)-1:
        y = np.moveaxis(y, idx_ax, -1)
    if overflow:
        # stupid, but if the mt has overflow even if low range is selected we get back an overflow bin with zeros 
        #   (even when we use the hist slicing methods), strip them off
        y = y[...,:-1]

    if use_weights:
        if hFRF.storage_type == hist.storage.Weight:
            # transform with weights
            w = 1./np.sqrt(hFRF.variances(flow=True))
            if idx_ax != len(hFRF.axes)-1:
                w = np.moveaxis(w, idx_ax, -1)
            if overflow:
                w = w[...,:-1]
        else:
            raise RuntimeError("Try to compute weighted least square but the histogram has no variances.")
    else:
        logger.warning("Using extendedABCD on histogram without uncertainties, make an unweighted linear squared solution.")
        w = np.ones_like(y)

    do1D=axis_name_2 is None
    stackX=[]   # parameter matrix X 
    stackXTY=[] # and X.T @ Y
    if do1D:
        logger.debug(f"Do the fakerate in 1D: {axis_name}")
        x = hFRF.axes[axis_name].centers
        x = np.broadcast_to(x, y.shape)
        for n in range(order+1):
            stackX.append(w*x**n)
            stackXTY.append((w**2 * x**n * y).sum(axis=(-1)))

        frf = lambda x, ps, o=order: sum([ps[...,n,np.newaxis] * x**n for n in range(o+1)])
    else:
        logger.debug(f"Do the fakerate in 2D: {axis_name},{axis_name_2}")

        idx_ax_2 = hFRF.axes.name.index(axis_name_2)
        if idx_ax_2 != len(hFRF.axes)-2:
            y = np.moveaxis(y, idx_ax_2, -2)
            w = np.moveaxis(w, idx_ax_2, -2)

        x = hFRF.axes[axis_name].centers
        x_2 = hFRF.axes[axis_name_2].centers

        x, x_2 = np.broadcast_arrays(x[np.newaxis,...], x_2[..., np.newaxis])

        x = np.broadcast_to(x, y.shape)
        x_2 = np.broadcast_to(x_2, y.shape)

        if order_2 is None:
            order_2=[0,]*(order+1)

        for n in range(order+1):
            for m in range(order_2[n]+1):
                stackX.append(w * x**n * x_2**m)
                stackXTY.append((w**2 * x**n * x_2**m * y).sum(axis=(-2, -1)))

        def frf(x1, x2, ps, o1=order, o2=order_2):
            idx=0
            FRF = 0
            for n in range(o1+1):
                for m in range(o2[n]+1):
                    FRF += ps[...,idx,np.newaxis,np.newaxis] * x1**n * x2**m
                    idx += 1
            return FRF

    X = np.stack(stackX, axis=-1)
    XTY = np.stack(stackXTY, axis=-1)
    
    if not do1D:
        # flatten the 2D array into 1D
        newshape = (*y.shape[:-2],np.product(y.shape[-2:]))
        y = y.reshape(newshape)
        X = X.reshape(*newshape, X.shape[-1])

    # compute the transpose of X for the mt and parameter axes 
    XT = np.transpose(X, axes=(*range(len(X.shape)-2), len(X.shape)-1, len(X.shape)-2))
    XTX = XT @ X
    # compute the inverse of the matrix in each bin (reshape to make last two axes contiguous, reshape back after inversion), 
    # this term is also the covariance matrix for the parameters
    XTXinv = np.linalg.inv(XTX.reshape(-1,*XTX.shape[-2:]))
    XTXinv = XTXinv.reshape((*XT.shape[:-2],*XTXinv.shape[-2:])) 
    params = np.einsum('...ij,...j->...i', XTXinv, XTY)

    if auxiliary_info:
        # goodness of fit
        residuals = y - np.einsum('...ij,...j->...i', X, params)
        chi2 = np.sum((residuals**2), axis=-1) # per fit chi2
        chi2_total = np.sum((residuals**2)) # chi2 of all fits together

        # Degrees of freedom calculation
        ndf = y.shape[-1] - params.shape[-1]
        ndf_total = y.size - params.size

        logger.info(f"Total chi2/ndf = {chi2_total}/{ndf_total} = {chi2_total/ndf_total}")

        return params, XTXinv, frf, chi2, ndf
    else:
        return params, XTXinv, frf


def fakeHistExtendedABCD(h, fakerate_integration_axes=[], fakerate_axis="mt", fakerate_bins=[0,4,11,21,40], 
    integrateLowMT=True, integrateHighMT=True, order=2, container=None, axis_name_2="pt", order_2=[2,1,2]):
    logger.debug(f"Compute fakes with extended ABCD method of order {order}")
    # fakerate_axis: The axis to parameterize the fakerate in (e.g. transverse mass)
    # fakerate_bins: The lower bins to solve the parameterization (e.g. transverse mass bins up to 40)
    # integrate=False keeps the fakerate axis in the returned histogram (can be used to have fakes vs transverse mass)
    # order: set the order for the parameterization: 0: f(x)=a; 1: f(x)=a+b*x; 2: f(x)=a+b*x+c*x**2; ...
    # container: Specify a dictionary to store the histogram used in the first call of this function and use it in all succeeding calls 
    #    (to use the variances of the nominal histogram when processing systematics)

    # allow for a parameterization in the fakerate factor (FRF) with fakerate axis (x)
    # 1) calculate FRF(x) in each bin of the fakerate axis
    # 2) obtain parameters from fakerate bins using weighted least square https://en.wikipedia.org/wiki/Weighted_least_squares
    # 3) apply the FRF(x) in high bins of fakertae axis

    if h.axes[fakerate_axis].traits.underflow:
        raise NotImplementedError(f"Underflow bins for axis {fakerate_axis} is not supported")

    overflow = h.axes[fakerate_axis].traits.overflow

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [n for n in h.axes.name if n not in fakerate_integration_axes]
        logger.info(f"Project {fakerate_axes}")
        h = h.project(*fakerate_axes)

    hLow = hh.rebinHist(h, fakerate_axis, fakerate_bins)

    if container is not None:
        if "h_fakerate_low" in container:
            # take variances from containered histogram
            hLow_2 = container["h_fakerate_low"]

            val_2 = hLow_2.values(flow=True)
            var_2 = hLow_2.variances(flow=True)
            val_1 = hLow.values(flow=True)
            # scale uncertainty with difference in yield with respect to second histogram such that relative uncertainty stays the same
            extra_dimensions = (Ellipsis,) + (np.newaxis,) * (val_1.ndim - val_2.ndim)
            var_1 = var_2[extra_dimensions] * (val_1 / val_2[extra_dimensions])**2

            hLow = hist.Hist(*hLow.axes, storage=hist.storage.Weight())
            hLow.view(flow=True)[...] = np.stack((val_1, var_1), axis=-1)
        else:
            # store histogram for later
            container["h_fakerate_low"] = hLow

    params, cov, frf = compute_fakerate(hLow, fakerate_axis, overflow, axis_name_2=axis_name_2, order_2=order_2, use_weights=True, order=order)

    hHighFail = h[{**common.failIso, fakerate_axis: slice(complex(0,fakerate_bins[-1]),None)}]

    val_c = hHighFail.values(flow=True)
    idx_ax = hHighFail.axes.name.index(fakerate_axis)
    if idx_ax != len(hHighFail.axes)-1:
        val_c = np.moveaxis(val_c, idx_ax, -1)

    x_c = hHighFail.axes[fakerate_axis].centers
    if overflow:
        # need an x point for the overflow bin, mirror distance between two last bins 
        x_c = np.append(x_c, x_c[-1]+np.diff(x_c[-2:]))

    if axis_name_2:
        x_2 = hHighFail.axes[axis_name_2].centers
        idx_ax_2 = hHighFail.axes.name.index(axis_name_2)
        if idx_ax_2 != len(hHighFail.axes)-2:
            val_c = np.moveaxis(val_c, idx_ax_2, -2)
        x_c, x_2 = np.broadcast_arrays(x_c[np.newaxis,...], x_2[..., np.newaxis])
        x_c = np.broadcast_to(x_c, val_c.shape)
        x_2 = np.broadcast_to(x_2, val_c.shape)
        FRF = frf(x_c, x_2, params)
        if idx_ax_2 != len(hHighFail.axes)-2:
            FRF = np.moveaxis(FRF, -2, idx_ax_2)
            val_c = np.moveaxis(val_c, -2, idx_ax_2)
    else:
        x_c = np.broadcast_to(x_c, val_c.shape)
        FRF = frf(x_c, params)

    val_d = val_c * FRF

    if h.storage_type == hist.storage.Weight:
        # variance coming from statistical uncertainty in C (indipendent bin errors, sum of squares)
        var_c = hHighFail.variances(flow=True)
        if idx_ax != len(hHighFail.axes)-1:
            var_c = np.moveaxis(var_c, idx_ax, -1)
        var_d = (FRF**2 * var_c).sum(axis=-1)

    if integrateHighMT:
        hHighPass = hist.Hist(*[a for a in hHighFail.axes if a.name != fakerate_axis], storage=h.storage_type())

        if h.storage_type == hist.storage.Weight:
            if axis_name_2:
                # TODO: implement proper uncertainties for fakerate parameters
                pass
            else:
                # variance from error propagation formula from parameters (first sum bins, then multiply with parameter error)
                for n in range(order+1):
                    var_d += (val_c*x_c**n).sum(axis=-1)**2 * cov[...,n,n]
                    for m in range(max(0,n)):
                        var_d += 2*(val_c*x_c**n).sum(axis=-1)*(val_c*x_c**m).sum(axis=-1)*cov[...,n,m]
            val_d = np.stack((val_d.sum(axis=-1), var_d), axis=-1)
    else:
        hHighPass = hHighFail.copy()

        if idx_ax != len(hHighFail.axes)-1:
            val_d = np.moveaxis(val_d, -1, idx_ax) # move the axis back

        if h.storage_type == hist.storage.Weight:
            logger.warning(f"Using extendedABCD as function of {fakerate_axis} without integration of mt will give fakes with incorrect bin by bin uncertainties!")
            # TODO: implement proper uncertainties for fakerate parameters
            # Wrong but easy solution for the moment, just omit the sums
            # variance from error propagation formula from parameters
            var_d = (val_c*x_c)**2 * cov[...,1,1,np.newaxis] + (val_c)**2 * cov[...,0,0,np.newaxis] + 2*(val_c*x_c)*val_c*cov[...,1,0,np.newaxis] 
            for n in range(order+1):
                var_d += (val_c*x_c**n)**2 * cov[...,n,n]
                for m in range(max(0,n-1)):
                    var_d += 2*(val_c*x_c**n)*(val_c*x_c**m)*cov[...,n,m]
            if idx_ax != len(hHighFail.axes)-1:
                var_d = np.moveaxis(var_d, -1, idx_ax) # move the axis back
            val_d = np.stack((val_d, var_d), axis=-1)

    hHighPass.view(flow=True)[...] = val_d 

    return hHighPass


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
        passMT = s[high::hist.sum] if integrateHighMT else s[high:]
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
