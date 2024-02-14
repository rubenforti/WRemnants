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

def fakeHistABCD(h, fakerate_integration_axes=[], 
    fakerate_axis_x="mt", threshold_x=40.0, fakerate_axis_y="iso", threshold_y=0.15, integrateLow=True, integrateHigh=False):
    # integrateMT=False keeps the mT axis in the returned histogram (can be used to have fakes vs mT)

    nameX, failX, passX = get_axis_fallback_selection(h, threshold_x, fakerate_axis_x, integrateLow, integrateHigh)
    nameY, failY, passY = get_axis_fallback_selection(h, threshold_y, fakerate_axis_y, integrateLow, integrateHigh)

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [n for n in h.axes.name if n not in [*fakerate_integration_axes, nameX, nameY]]
        hA = h[{nameY: passY, nameX: failX}].project(*fakerate_axes)
        hB = h[{nameY: failY, nameX: failX}].project(*fakerate_axes)
    else:
        hA = h[{nameY: passY, nameX: failX}]
        hB = h[{nameY: failY, nameX: failX}]

    hFRF = hh.divideHists(hA, hB, cutoff=1, createNew=True)   

    return hh.multiplyHists(hFRF, h[{nameY: failY, nameX: passX}])

def compute_fakerate(hSideband, axis_name="mt", remove_underflow=False, remove_overflow=True, 
    name_y="passIso", fail_y=0, pass_y=1,
    use_weights=True, order=1, auxiliary_info=False,
    smoothing_axis_name=None, smoothing_order=None,
):
    logger.debug(f"Do the fakerate in order: {order}")
    # fakerate parameterized in one variable, mt, depending on order of polinomial: f(mt) = a + b*mt + c*mt**2 ... 
    hPass = hSideband[{name_y: pass_y}]
    hFail = hSideband[{name_y: fail_y}]
    hFRF = hh.divideHists(hPass, hFail, cutoff=0.1, createNew=True, flow=True)
    y = hFRF.values(flow=True)

    idx_ax = hFRF.axes.name.index(axis_name)
    if idx_ax != len(hFRF.axes)-1:
        y = np.moveaxis(y, idx_ax, -1)

    # stupid, but if the axis has overflow even if low range is selected we get back an overflow bin with zeros 
    #   (even when we use the hist slicing methods), strip them off
    remove_underflow = axis_name in ["iso", "dxy"]
    remove_overflow = axis_name in ["mt",]
    if remove_overflow:
        y = y[...,:-1]
    if remove_underflow:
        y = y[...,1:]

    if use_weights:
        if hFRF.storage_type == hist.storage.Weight:
            # transform with weights
            w = 1./np.sqrt(hFRF.variances(flow=True))
            if idx_ax != len(hFRF.axes)-1:
                w = np.moveaxis(w, idx_ax, -1)
            if remove_overflow:
                w = w[...,:-1]
            if remove_underflow:
                w = w[...,1:]
        else:
            raise RuntimeError("Try to compute weighted least square but the histogram has no variances.")
    else:
        logger.warning("Using extendedABCD on histogram without uncertainties, make an unweighted linear squared solution.")
        w = np.ones_like(y)

    do1D=smoothing_axis_name is None
    stackX=[]   # parameter matrix X 
    stackXTY=[] # and X.T @ Y

    x = hFRF.axes[axis_name].centers
    if hFRF.axes[axis_name].traits.underflow and not remove_underflow:
        x = np.array([x[0]-np.diff(x[:2])])
    if hFRF.axes[axis_name].traits.overflow and not remove_overflow:
        x = np.append(x, x[-1]+np.diff(x[-2:]))

    if do1D:
        logger.debug(f"Do the fakerate in 1D: {axis_name}")
        x = np.broadcast_to(x, y.shape)
        for n in range(order+1):
            stackX.append(w*x**n)
            stackXTY.append((w**2 * x**n * y).sum(axis=(-1)))

        frf = lambda x, ps, o=order: sum([ps[...,n,np.newaxis] * x**n for n in range(o+1)])
    else:
        logger.debug(f"Do the fakerate in 2D: {axis_name},{smoothing_axis_name}")

        idx_ax_2 = hFRF.axes.name.index(smoothing_axis_name)
        if idx_ax_2 != len(hFRF.axes)-2:
            y = np.moveaxis(y, idx_ax_2, -2)
            w = np.moveaxis(w, idx_ax_2, -2)

        x_2 = hFRF.axes[smoothing_axis_name].centers

        x, x_2 = np.broadcast_arrays(x[np.newaxis,...], x_2[..., np.newaxis])

        x = np.broadcast_to(x, y.shape)
        x_2 = np.broadcast_to(x_2, y.shape)

        if smoothing_order is None:
            smoothing_order = [0,]*(order+1)

        for n in range(order+1):
            for m in range(smoothing_order[n]+1):
                stackX.append(w * x**n * x_2**m)
                stackXTY.append((w**2 * x**n * x_2**m * y).sum(axis=(-2, -1)))

        def frf(x1, x2, ps, o1=order, o2=smoothing_order):
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


def fakeHistFullExtendedABCD(h, fakerate_integration_axes=[], 
    fakerate_axis_x="mt", fakerate_bins_x=[0, 21, 40], 
    fakerate_axis_y="iso", fakerate_bins_y=[0, 0.15, 0.3], 
    order=2, integrateLow=True, integrateHigh=True, container=None, 
    smoothing_axis_name="pt", smoothing_order=[2,1,2]
):
    # full extended ABCD method as desribed in https://arxiv.org/abs/1906.10831 equation 15
    # add additional regions in x and in y

    flow=True

    # define sideband ranges
    dx = slice(complex(0, fakerate_bins_x[1]), complex(0, fakerate_bins_x[2]), hist.sum)
    dy = slice(complex(0, fakerate_bins_y[1]), complex(0, fakerate_bins_y[2]), hist.sum)

    s = hist.tag.Slicer()
    if fakerate_axis_x in ["mt"]:
        x = slice(complex(0, fakerate_bins_x[2]), None, hist.sum)
        d2x = slice(complex(0, fakerate_bins_x[0]), complex(0, fakerate_bins_x[1]), hist.sum)
    elif fakerate_axis_x in ["dxy", "iso"]:
        x = slice(complex(0, fakerate_bins_x[0]), complex(0, fakerate_bins_x[1]), hist.sum)
        d2x = slice(complex(0, fakerate_bins_x[2]), None, hist.sum)

    if fakerate_axis_y in ["mt"]:
        y = slice(complex(0, fakerate_bins_y[2]), None, hist.sum)
        d2y = slice(complex(0, fakerate_bins_y[0]), complex(0, fakerate_bins_y[1]), hist.sum)
    elif fakerate_axis_y in ["dxy", "iso"]:
        y = slice(complex(0, fakerate_bins_y[0]), complex(0, fakerate_bins_y[1]), hist.sum)
        d2y = slice(complex(0, fakerate_bins_y[2]), None, hist.sum)

    # select sideband regions
    a = h[{fakerate_axis_x: dx, fakerate_axis_y: dy}].values(flow=flow)
    b = h[{fakerate_axis_x: dx, fakerate_axis_y: y}].values(flow=flow)
    c = h[{fakerate_axis_x: x, fakerate_axis_y: dy}].values(flow=flow)
    bx = h[{fakerate_axis_x: d2x, fakerate_axis_y: y}].values(flow=flow)
    cy = h[{fakerate_axis_x: x, fakerate_axis_y: d2y}].values(flow=flow)
    ax = h[{fakerate_axis_x: d2x, fakerate_axis_y: dy}].values(flow=flow)
    ay = h[{fakerate_axis_x: dx, fakerate_axis_y: d2y}].values(flow=flow)
    axy = h[{fakerate_axis_x: d2x, fakerate_axis_y: d2y}].values(flow=flow)

    d = (ax*ay)**2/axy * (b*c)**2/a**4 * 1/(bx*cy)

    # set histogram in signal region
    hSignal = hist.Hist(*[a for a in h.axes if a.name not in [fakerate_axis_x, fakerate_axis_y] ], storage=h.storage_type())
    hSignal.values(flow=flow)[...] = d

    if h.storage_type == hist.storage.Weight:
        avar = h[{fakerate_axis_x: dx, fakerate_axis_y: dy}].variances(flow=flow)
        bvar = h[{fakerate_axis_x: dx, fakerate_axis_y: y}].variances(flow=flow)
        cvar = h[{fakerate_axis_x: x, fakerate_axis_y: dy}].variances(flow=flow)
        bxvar = h[{fakerate_axis_x: d2x, fakerate_axis_y: y}].variances(flow=flow)
        cyvar = h[{fakerate_axis_x: x, fakerate_axis_y: d2y}].variances(flow=flow)
        axvar = h[{fakerate_axis_x: d2x, fakerate_axis_y: dy}].variances(flow=flow)
        ayvar = h[{fakerate_axis_x: dx, fakerate_axis_y: d2y}].variances(flow=flow)
        axyvar = h[{fakerate_axis_x: d2x, fakerate_axis_y: d2y}].variances(flow=flow)

        dvar = d**2 * (4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 4*cvar/c**2 + 16*avar/a**2 + bxvar/bx**2 + cyvar/cy**2 )

        hSignal.variances(flow=flow)[...] = dvar

    return hSignal

def fakeHistExtendedABCD(h, 
    fakerate_integration_axes=[], fakerate_axis_x="mt", fakerate_bins_x=[0,4,11,21,40], order=2, integrateLow=True, integrateHigh=True, 
    fakerate_axis_y="passIso", threshold_y=None, 
    container=None, 
    smoothing_axis_name="pt", smoothing_order=[2,1,2]
):
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

    if h.axes[fakerate_axis_x].traits.underflow:
        raise NotImplementedError(f"Underflow bins for axis {fakerate_axis_x} is not supported")

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [n for n in h.axes.name if n not in fakerate_integration_axes]
        logger.info(f"Project {fakerate_axes}")
        h = h.project(*fakerate_axes)

    hSideband = hh.rebinHist(h, fakerate_axis_x, fakerate_bins_x)

    if container is not None:
        if "h_fakerate" in container:
            # take variances from containered histogram
            hSideband_0 = container["h_fakerate"]

            val_0 = hSideband_0.values(flow=True)
            var_0 = hSideband_0.variances(flow=True)
            val_1 = hSideband.values(flow=True)
            # scale uncertainty with difference in yield with respect to second histogram such that relative uncertainty stays the same
            extra_dimensions = (Ellipsis,) + (np.newaxis,) * (val_1.ndim - val_0.ndim)
            var_1 = var_0[extra_dimensions] * (val_1 / val_0[extra_dimensions])**2

            hSideband = hist.Hist(*hSideband.axes, storage=hist.storage.Weight())
            hSideband.view(flow=True)[...] = np.stack((val_1, var_1), axis=-1)
        else:
            # store histogram for later
            container["h_fakerate"] = hSideband

    if fakerate_axis_y in ["mt",] or fakerate_axis_y.startswith("pass"):
        fail_y, pass_y = get_axis_selection(h, threshold_y, fakerate_axis_y)
    elif fakerate_axis_y in ["dxy","iso"]:
        pass_y, fail_y = get_axis_selection(h, threshold_y, fakerate_axis_y)

    params, cov, frf = compute_fakerate(hSideband, fakerate_axis_x, name_y=fakerate_axis_y, pass_y=pass_y, fail_y=fail_y, 
        smoothing_axis_name=smoothing_axis_name, smoothing_order=smoothing_order, use_weights=False, order=order)

    # construct histogram in application region
    if fakerate_bins_x[0] == 0:
        hApplication = h[{fakerate_axis_y: fail_y, fakerate_axis_x: slice(complex(0,fakerate_bins_x[-1]),None)}]
        flow = h.axes[fakerate_axis_x].traits.overflow
    else:
        hApplication = h[{fakerate_axis_y: fail_y, fakerate_axis_x: slice(None, complex(0, fakerate_bins_x[0]))}]
        flow = False

    val_c = hApplication.values(flow=flow)
    idx_ax = hApplication.axes.name.index(fakerate_axis_x)
    if idx_ax != len(hApplication.axes)-1:
        val_c = np.moveaxis(val_c, idx_ax, -1)

    x_c = hApplication.axes[fakerate_axis_x].centers
    if flow:
        # need an x point for the overflow bin, mirror distance between two last bins 
        x_c = np.append(x_c, x_c[-1]+np.diff(x_c[-2:]))

    if smoothing_axis_name:
        x_smoothing = hApplication.axes[smoothing_axis_name].centers
        idx_ax_smoothing = hApplication.axes.name.index(smoothing_axis_name)
        if idx_ax_smoothing != len(hApplication.axes)-2:
            val_c = np.moveaxis(val_c, idx_ax_smoothing, -2)
        x_c, x_smoothing = np.broadcast_arrays(x_c[np.newaxis,...], x_smoothing[..., np.newaxis])
        x_c = np.broadcast_to(x_c, val_c.shape)
        x_smoothing = np.broadcast_to(x_smoothing, val_c.shape)
        FRF = frf(x_c, x_smoothing, params)
        if idx_ax_smoothing != len(hApplication.axes)-2:
            FRF = np.moveaxis(FRF, -2, idx_ax_smoothing)
            val_c = np.moveaxis(val_c, -2, idx_ax_smoothing)
    else:
        x_c = np.broadcast_to(x_c, val_c.shape)
        FRF = frf(x_c, params)

    val_d = val_c * FRF

    if h.storage_type == hist.storage.Weight:
        # variance coming from statistical uncertainty in C (indipendent bin errors, sum of squares)
        var_c = hApplication.variances(flow=flow)
        if idx_ax != len(hApplication.axes)-1:
            var_c = np.moveaxis(var_c, idx_ax, -1)
        var_d = (FRF**2 * var_c).sum(axis=-1)

    # set the histogram in the signal region
    if integrate:
        hSignal = hist.Hist(*[a for a in hApplication.axes if a.name != fakerate_axis_x], storage=h.storage_type())
        val_d = val_d.sum(axis=-1)
        if h.storage_type == hist.storage.Weight:
            if smoothing_axis_name:
                # TODO: implement proper uncertainties for fakerate parameters in case of smoothing
                pass
            else:
                # variance from error propagation formula from parameters (first sum bins, then multiply with parameter error)
                for n in range(order+1):
                    var_d += (val_c*x_c**n).sum(axis=-1)**2 * cov[...,n,n]
                    for m in range(max(0,n)):
                        var_d += 2*(val_c*x_c**n).sum(axis=-1)*(val_c*x_c**m).sum(axis=-1)*cov[...,n,m]
            val_d = np.stack((val_d, var_d), axis=-1)            
    else:
        hSignal = hApplication.copy()
        if idx_ax != len(hApplication.axes)-1:
            val_d = np.moveaxis(val_d, -1, idx_ax) # move the axis back
        if h.storage_type == hist.storage.Weight:
            logger.warning(f"Using extendedABCD as function of {fakerate_axis_x} without integration of mt will give fakes with incorrect bin by bin uncertainties!")
            # TODO: implement proper uncertainties for fakerate parameters
            # Wrong but easy solution for the moment, just omit the sums
            # variance from error propagation formula from parameters
            var_d = (val_c*x_c)**2 * cov[...,1,1,np.newaxis] + (val_c)**2 * cov[...,0,0,np.newaxis] + 2*(val_c*x_c)*val_c*cov[...,1,0,np.newaxis] 
            for n in range(order+1):
                var_d += (val_c*x_c**n)**2 * cov[...,n,n]
                for m in range(max(0,n-1)):
                    var_d += 2*(val_c*x_c**n)*(val_c*x_c**m)*cov[...,n,m]
            if idx_ax != len(hApplication.axes)-1:
                var_d = np.moveaxis(var_d, -1, idx_ax) # move the axis back
            val_d = np.stack((val_d, var_d), axis=-1)

    hSignal.view(flow=flow)[...] = val_d 

    return hSignal


def fakeHistSimultaneousABCD(h, threshold_x=40.0, fakerate_integration_axes=[], axis_name_x="mt", integrateLow=True, integrateHigh=False
):
    if h.storage_type == hist.storage.Weight:
        # setting errors to 0
        h.view(flow=True)[...] = np.stack((h.values(flow=True), np.zeros_like(h.values(flow=True))), axis=-1)

    nameMT, failMT, passMT = get_mt_selection(h, threshold_x, axis_name_x, integrateLow, integrateHigh)

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
    h.values(flow=True)[*slices] = fakeHistABCD(h, threshold, fakerate_integration_axes, axis_name_x, integrateLow, integrateHigh).values(flow=True)

    return h

def signalHistWmass(h, axis_name_x="mt", threshold_x=40.0, axis_name_y="iso", threshold_y=0.15, integrateLow=True, integrateHigh=True):
    nameX, failX, passX = get_axis_fallback_selection(h, threshold_x, axis_name_x, integrateLow, integrateHigh)
    nameY, failY, passY = get_axis_fallback_selection(h, threshold_y, axis_name_y, integrateLow, integrateHigh)
    return h[{nameX: passX, nameY: passY}]

def get_axis_fallback_selection(h, threshold=40.0, axis_name="mt", integrateLow=True, integrateHigh=False):
    if axis_name in h.axes.name:
        name = axis_name
    elif axis_name=="mt":
        name = common.passMTName  
    elif axis_name=="iso":
        name = common.passIsoName  
    else:
        raise NotImplementedError(f"No fallback selection for {axis_name} implemented")
    lo, hi = get_axis_selection(h, threshold, name, integrateLow, integrateHigh)

    # return values in correct order: name, fail, pass
    if axis_name=="mt" or name.startswith("pass"):
        return name, lo, hi
    else:
        return name, hi, lo

def get_axis_selection(h, threshold=0.15, axis_name="iso", integrateLow=True, integrateHigh=True):
    if type(h.axes[axis_name]) == hist.axis.Boolean:
        lo = 0
        hi = 1
    else:
        s = hist.tag.Slicer()
        high = h.axes[axis_name].index(threshold)
        lo = s[:high:hist.sum] if integrateLow else s[:high:]
        hi = s[high::hist.sum] if integrateHigh else s[high::]
    return lo, hi

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