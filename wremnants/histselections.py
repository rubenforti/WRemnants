import hist
import numpy as np
from utilities import boostHistHelpers as hh
from utilities import common, logging

from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import unumpy as unp

import pdb

logger = logging.child_logger(__name__)

thresholds={
    "mt":40,
    "iso":0.15,
    "dxy":0.01
}

def exp_fall(x, a, b, c):
    return a * np.exp(-b * x) + c
def exp_fall_unc(x, a, b, c):
    return a * unp.exp(-b * x) + c

def get_abcd_axes(h):
    axis_x=None
    axis_y=None
    for a, b in (("mt", "passMT"), ("iso", "passIso"), ("dxy", "passDxy")):
        if axis_x is None and a in h.axes.name: 
            axis_x=a
            continue
        if axis_x is None and b in h.axes.name: 
            axis_x=b
            continue            
        if axis_y is None and a in h.axes.name: 
            axis_y=a
            continue
        if axis_y is None and b in h.axes.name: 
            axis_y=b
            continue
    if axis_y is None:
        raise RuntimeError(f"Can not find ABCD axes among histogram axes {h.axes.name}")
    else:
        logger.debug(f"Found ABCD axes {axis_x} and {axis_y}")
    return axis_x, axis_y

def get_abcd_selection(h, axis_name, integrateLow=True, integrateHigh=True):
    if type(h.axes[axis_name]) == hist.axis.Boolean:
        return 0, 1
    if axis_name not in thresholds:
        raise RuntimeError(f"Can not find threshold for abcd axis {axis_name}")
    tBin = h.axes[axis_name].index(thresholds[axis_name])
    s = hist.tag.Slicer()
    lo = s[:tBin:hist.sum] if integrateLow else s[:tBin:]
    hi = s[tBin::hist.sum] if integrateHigh else s[tBin::]
    if axis_name in ["mt",]:
        return lo, hi
    else:
        return hi, lo

def get_abcd_bins(h, axis_name):
    if axis_name not in thresholds:
        raise RuntimeError(f"Can not find threshold for abcd axis {axis_name}")
    f, _ = get_abcd_selection(h, axis_name, integrateLow=False, integrateHigh=False)
    return h[{axis_name: f}].axes[axis_name].edges
    

# signal region selection
def signalHistABCD(h, integrateLow=True, integrateHigh=True):
    nameX, nameY = get_abcd_axes(h)
    failX, passX = get_abcd_selection(h, nameX, integrateLow, integrateHigh)
    failY, passY = get_abcd_selection(h, nameY, integrateLow, integrateHigh)
    return h[{nameX: passX, nameY: passY}]

# simple ABCD method
def fakeHistABCD(h, fakerate_integration_axes=[], integrateLow=True, integrateHigh=True):
    # integrateMT=False keeps the mT axis in the returned histogram (can be used to have fakes vs mT)

    nameX, nameY = get_abcd_axes(h)
    failX, passX = get_abcd_selection(h, nameX, integrateLow, integrateHigh)
    failY, passY = get_abcd_selection(h, nameY, integrateLow, integrateHigh)

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [n for n in h.axes.name if n not in [*fakerate_integration_axes, nameX, nameY]]
        hA = h[{nameY: passY, nameX: failX}].project(*fakerate_axes)
        hB = h[{nameY: failY, nameX: failX}].project(*fakerate_axes)
    else:
        hA = h[{nameY: passY, nameX: failX}]
        hB = h[{nameY: failY, nameX: failX}]

    hFRF = hh.divideHists(hA, hB, cutoff=1, createNew=True)   

    return hh.multiplyHists(hFRF, h[{nameY: failY, nameX: passX}])

def get_parameter_matrices(x, y, w, order, mask=None):
    if x.shape != y.shape:
        x = np.broadcast_to(x, y.shape)
    stackX=[] # parameter matrix X 
    stackXTY=[] # and X.T @ Y
    for n in range(order+1):
        stackX.append(w*x**n)
        stackXTY.append((w**2 * x**n * y).sum(axis=(-1)))
    f = lambda x, ps, o=order: sum([ps[...,n,np.newaxis] * x**n for n in range(o+1)])
    X = np.stack(stackX, axis=-1)
    XTY = np.stack(stackXTY, axis=-1)
    return X, XTY, f

def solve_leastsquare(X, XTY):
    # compute the transpose of X for the mt and parameter axes 
    XT = np.transpose(X, axes=(*np.arange(X.ndim-2), X.ndim-1, X.ndim-2))
    XTX = XT @ X
    # compute the inverse of the matrix in each bin (reshape to make last two axes contiguous, reshape back after inversion), 
    # this term is also the covariance matrix for the parameters
    XTXinv = np.linalg.inv(XTX.reshape(-1,*XTX.shape[-2:]))
    XTXinv = XTXinv.reshape((*XT.shape[:-2],*XTXinv.shape[-2:])) 
    params = np.einsum('...ij,...j->...i', XTXinv, XTY)
    return params, XTXinv

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


    x = hFRF.axes[axis_name].centers
    if hFRF.axes[axis_name].traits.underflow and not remove_underflow:
        x = np.array([x[0]-np.diff(x[:2])])
    if hFRF.axes[axis_name].traits.overflow and not remove_overflow:
        x = np.append(x, x[-1]+np.diff(x[-2:]))

    if do1D:
        logger.debug(f"Do the fakerate in 1D: {axis_name}")
        X, XTY, frf = get_parameter_matrices(x, y, w, order)
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

        stackX=[]   # parameter matrix X 
        stackXTY=[] # and X.T @ Y
        for n in range(order+1):
            for m in range(smoothing_order[n]+1):
                stackX.append(w * x**n * x_2**m)
                stackXTY.append((w**2 * x**n * x_2**m * y).sum(axis=(-2, -1)))
        X = np.stack(stackX, axis=-1)
        XTY = np.stack(stackXTY, axis=-1)

        def frf(x1, x2, ps, o1=order, o2=smoothing_order):
            idx=0
            f = 0
            for n in range(o1+1):
                for m in range(o2[n]+1):
                    f += ps[...,idx,np.newaxis,np.newaxis] * x1**n * x2**m
                    idx += 1
            return f
    
    if not do1D:
        # flatten the 2D array into 1D
        newshape = (*y.shape[:-2],np.product(y.shape[-2:]))
        y = y.reshape(newshape)
        X = X.reshape(*newshape, X.shape[-1])

    params, cov = solve_leastsquare(X, XTY)    

    if auxiliary_info:
        # goodness of fit
        residuals = y - np.einsum('...ij,...j->...i', X, params)
        chi2 = np.sum((residuals**2), axis=-1) # per fit chi2
        chi2_total = np.sum((residuals**2)) # chi2 of all fits together

        # Degrees of freedom calculation
        ndf = y.shape[-1] - params.shape[-1]
        ndf_total = y.size - params.size

        logger.info(f"Total chi2/ndf = {chi2_total}/{ndf_total} = {chi2_total/ndf_total}")

        return params, cov, frf, chi2, ndf
    else:
        return params, cov, frf

# extended ABCD with extra regions in the ABCD x-axis
def fakeHistExtendedABCD(h, 
    fakerate_integration_axes=[], order=2, integrateLow=True, integrateHigh=True, 
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

    nameX, nameY = get_abcd_axes(h)
    binsX = get_abcd_bins(h, nameX)
    failY, passY = get_abcd_selection(h, nameY)

    if h.axes[nameX].traits.underflow:
        raise NotImplementedError(f"Underflow bins for axis {nameX} is not supported")

    if any(a in h.axes.name for a in fakerate_integration_axes):
        fakerate_axes = [n for n in h.axes.name if n not in fakerate_integration_axes]
        logger.info(f"Project {fakerate_axes}")
        h = h.project(*fakerate_axes)

    hSideband = hh.rebinHist(h, nameX, binsX)

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

    params, cov, frf = compute_fakerate(hSideband, nameX, name_y=nameY, pass_y=passY, fail_y=failY, 
        smoothing_axis_name=smoothing_axis_name, smoothing_order=smoothing_order, use_weights=True, order=order)

    # construct histogram in application region
    if binsX[0] == 0:
        hApplication = h[{nameY: failY, nameX: slice(complex(0,binsX[-1]),None)}]
        flow = h.axes[nameX].traits.overflow
    else:
        hApplication = h[{nameY: failY, nameX: slice(None, complex(0, binsX[0]))}]
        flow = False

    val_c = hApplication.values(flow=flow)
    idx_ax = hApplication.axes.name.index(nameX)
    if idx_ax != len(hApplication.axes)-1:
        val_c = np.moveaxis(val_c, idx_ax, -1)

    x_c = hApplication.axes[nameX].centers

    if flow:
        # need an x point for the overflow bin, mirror distance between two last bins 
        x_all = h.axes[nameX].centers # use full range in case only overflow bin in application region
        x_c = np.append(x_c, x_all[-1]+np.diff(x_all[-2:]))

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
    if integrateLow and integrateHigh:
        hSignal = hist.Hist(*[a for a in hApplication.axes if a.name != nameX], storage=h.storage_type())
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
            logger.warning(f"Using extendedABCD as function of {nameX} without integration of mt will give fakes with incorrect bin by bin uncertainties!")
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

# full ABCD method as desribed in https://arxiv.org/abs/1906.10831 equation 15
def fakeHistFullABCD(h, fakerate_integration_axes=[], 
    order=2, integrateLow=True, integrateHigh=True, container=None, 
    # smoothing_axis_name="pt", smoothing_order=1
    smoothing_axis_name="", smoothing_order=1
):
    # add additional regions in x and in y
    nameX, nameY = get_abcd_axes(h)
    binsX = h.axes[nameX].edges
    binsY = h.axes[nameY].edges

    flow=True

    # define sideband ranges
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

    hSignal = hist.Hist(*[a for a in h.axes if a.name not in [nameX, nameY] ], storage=h.storage_type())

    # if smoothing_axis_name:
    #     logger.info("Make smooth Fakes")
    #     smoothing_axis_idx = h.axes.name.index(smoothing_axis_name)
    #     # if smoothing_axis_idx != len(h.axes):
    #     #     a = np.moveaxis(a, smoothing_axis_idx, -1)

    #     x_smooth = h.axes[smoothing_axis_name].centers
    #     # x_smooth = np.broadcast_to(x_smooth, a.shape)

    #     ds = []
    #     ds_var = []
    #     for iCharge, bCharge in enumerate(h.axes["charge"]):
    #         for iEta, bEta in enumerate(h.axes["eta"]):

    #             ibin = {"charge":iCharge, "eta":iEta}

    #             def fit(bin):
    #                 val = h[{**ibin, **bin}].values(flow=flow)
    #                 var = h[{**ibin, **bin}].variances(flow=flow)
    #                 # setting small numbers
    #                 val[val < 0] = 0
    #                 var[var < 1] = 1

    #                 params, cov = curve_fit(exp_fall, x_smooth, val, p0=[sum(val)/0.18, 0.18, min(val)], sigma=var**0.5, absolute_sigma=False)
    #                 params = unc.correlated_values(params, cov)
    #                 y_fit_u = exp_fall_unc(x_smooth, *params)
    #                 yy = np.array([y.n for y in y_fit_u])
    #                 yy_var = np.array([y.s**2 for y in y_fit_u])

    #                 return yy, yy_var

    #             # select sideband regions
    #             a, avar = fit({nameX: dx, nameY: dy})
    #             b, bvar = fit({nameX: dx, nameY: y})
    #             c, cvar = fit({nameX: x, nameY: dy})
    #             bx, bxvar = fit({nameX: d2x, nameY: y})
    #             cy, cyvar = fit({nameX: x, nameY: d2y})
    #             ax, axvar = fit({nameX: d2x, nameY: dy})
    #             ay, ayvar = fit({nameX: dx, nameY: d2y})
    #             axy, axyvar = fit({nameX: d2x, nameY: d2y})

    #             d = (ax*ay)**2/axy * (b*c)**2/a**4 * 1/(bx*cy)
    #             dvar = d**2 * (4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 4*cvar/c**2 + 16*avar/a**2 + bxvar/bx**2 + cyvar/cy**2 )

    #             slices = [slice(None)] * hSignal.ndim
    #             idx_eta = hSignal.axes.name.index("eta")
    #             idx_charge = hSignal.axes.name.index("charge")
    #             slices[idx_eta] = iEta
    #             slices[idx_charge] = iCharge

    #             hSignal.values(flow=True)[*slices] = d
    #             hSignal.variances(flow=True)[*slices] = dvar
    # else:
    
    # select sideband regions
    a = h[{nameX: dx, nameY: dy}].values(flow=flow)
    ax = h[{nameX: d2x, nameY: dy}].values(flow=flow)
    ay = h[{nameX: dx, nameY: d2y}].values(flow=flow)
    axy = h[{nameX: d2x, nameY: d2y}].values(flow=flow)
    b = h[{nameX: dx, nameY: y}].values(flow=flow)
    bx = h[{nameX: d2x, nameY: y}].values(flow=flow)

    c = h[{nameX: x, nameY: dy}].values(flow=flow)
    cy = h[{nameX: x, nameY: d2y}].values(flow=flow)

    if h.storage_type == hist.storage.Weight:
        avar = h[{nameX: dx, nameY: dy}].variances(flow=flow)
        axvar = h[{nameX: d2x, nameY: dy}].variances(flow=flow)
        ayvar = h[{nameX: dx, nameY: d2y}].variances(flow=flow)
        axyvar = h[{nameX: d2x, nameY: d2y}].variances(flow=flow)
        bvar = h[{nameX: dx, nameY: y}].variances(flow=flow)
        bxvar = h[{nameX: d2x, nameY: y}].variances(flow=flow)

        cvar = h[{nameX: x, nameY: dy}].variances(flow=flow)
        cyvar = h[{nameX: x, nameY: d2y}].variances(flow=flow)

    if smoothing_axis_name:
        # fakerate factor
        y_frf_num = (ax*ay*b)**2
        y_frf_den = a**4 * axy * bx
        y_frf = y_frf_num/y_frf_den
        mask_frf = (y_frf_den <= 0) # masking bins with negative entries
        if mask_frf.sum() > 0:
            logger.warning(f"{mask_frf.sum()} bins with negative or empty content found for denominator in the fakerate factor, these will be masked from the smoothing.")
        # transfer factor
        y_tf = c/cy
        mask_tf = (cy <= 0) # masking bins with negative entries
        if mask_tf.sum() > 0:
            logger.warning(f"{mask_tf.sum()} bins with negative or empty content found  for denominator in the transfer factor, these will be masked from the smoothing.")
        
        if h.storage_type == hist.storage.Weight:
            # masking bins with large statistical uncertainty (relative uncertainty > 100%)
            cut_rel_unc = 1
            y_frf_den_var = y_frf_den**2 * (16*avar/a**2 + axyvar/axy**2 + bxvar/bx**2)
            mask_frf_den = (y_frf_den_var**0.5/y_frf_den > cut_rel_unc)
            if mask_frf_den.sum() > 0:
                logger.warning(f"{mask_frf_den.sum()} bins with a relative uncertainty larger than {cut_rel_unc*100}% for denominator in the fakerate factor, these will be masked from the smoothing.")
            mask_frf = mask_frf | mask_frf_den

            mask_tf_den = (cyvar**0.5/cy > cut_rel_unc)
            if mask_tf_den.sum() > 0:
                logger.warning(f"{mask_tf_den.sum()} bins with a relative uncertainty larger than {cut_rel_unc*100}% for denominator in the transfer factor, these will be masked from the smoothing.")
            mask_tf = mask_frf | mask_tf_den

            # full variances
            y_frf_var = y_frf**2 * (4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 16*avar/a**2 + bxvar/bx**2)
            y_tf_var = y_tf**2 * (cvar/c**2 + cyvar/cy**2 )

            # transform with weights
            w_frf = 1/np.sqrt(y_frf_var)
            w_tf = 1/np.sqrt(y_tf_var)
        else:
            logger.warning("Smoothing extended ABCD on histogram without uncertainties, make an unweighted linear squared solution.")
            w_frf = np.ones_like(y_frf)
            w_tf = np.ones_like(y_tf)

        # smooth frf in pT
        x_smoothing = hSignal.axes[smoothing_axis_name].centers
        # move smoothing axis to last
        idx_ax_smoothing = hSignal.axes.name.index(smoothing_axis_name)
        if idx_ax_smoothing != len(hSignal.axes)-1:
            y_frf = np.moveaxis(y_frf, idx_ax_smoothing, -1)
            w_frf = np.moveaxis(w_frf, idx_ax_smoothing, -1)
            mask_frf = np.moveaxis(mask_frf, idx_ax_smoothing, -1)
            y_tf = np.moveaxis(y_tf, idx_ax_smoothing, -1)
            w_tf = np.moveaxis(w_tf, idx_ax_smoothing, -1)

            c = np.moveaxis(c, idx_ax_smoothing, -1)
            cy = np.moveaxis(cy, idx_ax_smoothing, -1)
            cvar = np.moveaxis(cvar, idx_ax_smoothing, -1)
            cyvar = np.moveaxis(cyvar, idx_ax_smoothing, -1)

        X_frf, XTY_frf, f_frf = get_parameter_matrices(x_smoothing, y_frf, w_frf, smoothing_order, mask_frf)
        X_tf, XTY_tf, f_tf = get_parameter_matrices(x_smoothing, y_tf, w_tf, smoothing_order, mask_tf)

        params_frf, cov_frf = solve_leastsquare(X_frf, XTY_frf)    
        params_tf, cov_tf = solve_leastsquare(X_tf, XTY_tf)    

        x_smooth = hSignal.axes[smoothing_axis_name].centers
        y_frf_smooth = f_frf(x_smooth, params_frf)
        y_tf_smooth = f_tf(x_smooth, params_tf)

        d = c * y_tf * y_frf_smooth
        dvar = d**2 * (4*cvar/c**2 + cyvar/cy**2) # stat uncertainty from application region

        # move smoothing axis to original positon again
        if idx_ax_smoothing != len(hSignal.axes)-1:
            d = np.moveaxis(d, -1, idx_ax_smoothing)
            dvar = np.moveaxis(dvar, -1, idx_ax_smoothing)
    else:
        # no smoothing
        d = (ax*ay)**2/axy * (b*c)**2/a**4 * 1/(bx*cy)
        dvar = d**2 * (4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 4*cvar/c**2 + 16*avar/a**2 + bxvar/bx**2 + cyvar/cy**2 )

    # set histogram in signal region
    hSignal.values(flow=flow)[...] = d
    hSignal.variances(flow=flow)[...] = dvar

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