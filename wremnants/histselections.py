import hist
import numpy as np
from utilities import boostHistHelpers as hh
from utilities import common, logging
from utilities.fnnls import fnnls
from scipy.optimize import nnls
from scipy.special import comb
import pdb

from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import unumpy as unp


logger = logging.child_logger(__name__)

abcd_thresholds={
    "mt":[0,20,40],
    "iso":[0,0.15,0.3,0.45],
    "dxy":[0,0.01,0.02,0.03]
}
# default abcd_variables to look for
abcd_variables = (("mt", "passMT"), ("iso", "passIso"), ("dxy", "passDxy"))

def get_eigen_variations(params, cov, sign=1, force_positive=True):
    # diagonalize and get eigenvalues and eigenvectors
    e, v = np.linalg.eigh(cov) # The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i], 
    vT = np.transpose(v, axes=(*np.arange(v.ndim-2), v.ndim-1, v.ndim-2)) # transpose v to have row eigenvectors[i, :] for easier computations
    mag = np.sqrt(e)[...,np.newaxis] * vT * sign
    # duplicate params vector to have duplicated rows
    params_brd = np.broadcast_to(params[...,np.newaxis], mag.shape)
    paramsT = np.transpose(params_brd, axes=(*np.arange(params_brd.ndim-2), params_brd.ndim-1, params_brd.ndim-2))
    mask = (paramsT + mag < 0).any(axis=-1)
    if force_positive and mask.sum() > 0:
        logger.info(f"Force {mask.sum()} eigenvectors to be positive")
        # scaling the magnitude of the eigenvector to avoid negative coefficients
        min_idxs = paramsT + mag == np.min(paramsT + mag, axis=-1)[...,np.newaxis] # find indices with minimum value of each eigenvector
        min_paramsT = paramsT[min_idxs]
        min_mag = mag[min_idxs]
        factors = np.reshape(-1*min_paramsT/min_mag, mask.shape)
        factors = np.where(mask, factors, np.ones_like(mask))
        factors = np.broadcast_to(factors[...,np.newaxis], mag.shape)
        mag = factors * mag
    if np.sum(mag.sum(axis=-1) == 0):
        logger.warning(f"Found {np.sum(mag.sum(axis=-1) == 0)} eigenvector shifts where all coefficients are 0") 
    params_var = paramsT + mag
    if np.sum(params_var.sum(axis=-1) == 0):
        logger.warning(f"Found {np.sum(params_var.sum(axis=-1) == 0)} variations where all coefficients are 0") 
    return params_var

def make_eigen_variations(params, cov, func, x1, x2=None, force_positive=False):
    params_up = get_eigen_variations(params, cov, sign=1, force_positive=force_positive)
    y_pred_up = func(x1, params_up) if x2 is None else func(x1, x2, params_up)
    y_pred_up = np.moveaxis(y_pred_up, params.ndim-1, -1) # put parameter variations last
    params_dn = get_eigen_variations(params, cov, sign=-1, force_positive=force_positive)
    y_pred_dn = func(x1, params_dn) if x2 is None else func(x1, x2, params_dn)
    y_pred_dn = np.moveaxis(y_pred_dn, params.ndim-1, -1) # put parameter variations last
    return np.stack((y_pred_up, y_pred_dn), axis=-1)

def get_parameter_matrices(x, y, w, order, pol="power"):
    if x.shape != y.shape:
        x = np.broadcast_to(x, y.shape)
    stackX=[] # parameter matrix X 
    stackXTY=[] # and X.T @ Y
    for n in range(order+1):
        if pol == "power":
            p = x**n
        elif pol=="bernstein":
            p = comb(order, n) * x**n * (1 - x)**(order - n)
        stackX.append( w * p )
        stackXTY.append((w**2 * p * y).sum(axis=(-1)))
    X = np.stack(stackX, axis=-1)
    XTY = np.stack(stackXTY, axis=-1)
    return X, XTY

def get_parameter_matrices_from2D(x, x2, y, w, order, order2=None, pol="power", flatten=False):
    x, x2 = np.broadcast_arrays(x[np.newaxis,...], x2[..., np.newaxis])
    x = np.broadcast_to(x, y.shape)
    x2 = np.broadcast_to(x2, y.shape)
    if order2 is None:
        order2 = [0,]*(order+1)
    elif type(order2) == int:
        order2 = [order2,]*(order+1)
    elif type(order2) == list: 
        if len(order2) < order+1:
            order2.append([0,]*(len(order2)-order+1))
    else:
        raise RuntimeError(f"Input 'order2' requires type 'None', 'int' or 'list'")
    stackX=[]   # parameter matrix X 
    stackXTY=[] # and X.T @ Y
    for n in range(order+1):
        for m in range(order2[n]+1):
            if pol=="power":
                p = x**n * x2**m
            elif pol=="bernstein":
                p = comb(order, n) * x**n * (1 - x)**(order - n) * comb(order2[n], m) * x2**m * (1 - x2)**(order2[n] - m)
            stackX.append(w * p)
            stackXTY.append((w**2 * p * y).sum(axis=(-2, -1)))
    X = np.stack(stackX, axis=-1)
    XTY = np.stack(stackXTY, axis=-1)
    if flatten:
        # flatten the 2D array into 1D
        newshape = (*y.shape[:-2],np.product(y.shape[-2:]))
        y = y.reshape(newshape)
        X = X.reshape(*newshape, X.shape[-1])

    return X, y, XTY

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

def solve_nonnegative_leastsquare(X, XTY):
    XT = np.transpose(X, axes=(*np.arange(X.ndim-2), X.ndim-1, X.ndim-2))
    XTX = XT @ X
    XTXinv = np.linalg.inv(XTX.reshape(-1,*XTX.shape[-2:]))
    XTXinv = XTXinv.reshape((*XT.shape[:-2],*XTXinv.shape[-2:])) 
    orig_shape = XTY.shape
    nBins = np.prod(orig_shape[:-1])
    XTY_flat = XTY.reshape(nBins, XTY.shape[-1])
    XTX_flat = XTX.reshape(nBins, XTX.shape[-2], XTX.shape[-1])
    # params = [fnnls(xtx, xty) for xtx, xty in zip(XTX_flat, XTY_flat)] # use fast nnls
    params = [nnls(xtx, xty)[0] for xtx, xty in zip(XTX_flat, XTY_flat)] # use scipy implementation of nnls (may be a bit slower)
    params = np.reshape(params, orig_shape)
    return params, XTXinv

def compute_chi2(y, y_pred, w=None, nparams=1):
    # goodness of fit from parameter matrix 'X', values 'y' and parateters 'params'
    residuals = (y - y_pred)
    if w is not None:
        residuals *= w
    chi2 = np.sum((residuals**2), axis=-1) # per fit chi2
    chi2_total = np.sum((residuals**2)) # chi2 of all fits together

    # Degrees of freedom calculation
    ndf = y.shape[-1] - nparams
    ndf_total = y.size - chi2.size*nparams

    logger.info(f"Total chi2/ndf = {chi2_total}/{ndf_total} = {chi2_total/ndf_total}")
    return chi2, ndf

def get_smoothing_function_1D(order, pol="power"):
    if pol=="power":
        def f(x, ps, o=order):
            if hasattr(ps, "ndim") and ps.ndim > 1:
                return sum([ps[...,n,np.newaxis] * x**n for n in range(o+1)])
            else:
                return sum([ps[n] * x**n for n in range(o+1)])
    elif pol=="bernstein":
        def f(x, ps, o=order):
            if hasattr(ps, "ndim") and ps.ndim > 1:
                return sum([ps[...,n,np.newaxis] * comb(o, n) * x**n * (1 - x)**(o - n) for n in range(o+1)])
            else:
                return sum([ps[n] * comb(o, n) * x**n * (1 - x)**(o - n) for n in range(o+1)])
    return f

def get_smoothing_function_2D(order1, order2, pol="power"):
    if pol=="power":
        def f(x1, x2, ps, o1=order1, o2=order2):
            idx=0
            f = 0
            x1, x2 = np.broadcast_arrays(x1[np.newaxis,...], x2[..., np.newaxis])
            if hasattr(ps, "ndim") and ps.ndim > 1:
                x1 = np.broadcast_to(x1, [*ps.shape[:-1], *x1.shape])
                x2 = np.broadcast_to(x2, [*ps.shape[:-1], *x2.shape]) 
                for n in range(o1+1):
                    for m in range(o2[n]+1):
                        f += ps[...,idx,np.newaxis,np.newaxis] * x1**n * x2**m
                        idx += 1
            else:
                for n in range(o1+1):
                    for m in range(o2[n]+1):
                        f += ps[idx] * x1**n * x2**m
                        idx += 1            
            return f
    elif pol=="bernstein":
        def f(x1, x2, ps, o1=order1, o2=order2):
            idx=0
            f = 0
            x1, x2 = np.broadcast_arrays(x1[np.newaxis,...], x2[..., np.newaxis])
            if hasattr(ps, "ndim") and ps.ndim > 1:
                x1 = np.broadcast_to(x1, [*ps.shape[:-1], *x1.shape])
                x2 = np.broadcast_to(x2, [*ps.shape[:-1], *x2.shape]) 
                for n in range(o1+1):
                    for m in range(o2[n]+1):
                        f += ps[...,idx,np.newaxis,np.newaxis] * comb(o1, n) * x1**n * (1 - x1)**(o1 - n) * comb(o2[n], m) * x2**m * (1 - x2)**(o2[n] - m)
                        idx += 1
            else:
                for n in range(o1+1):
                    for m in range(o2[n]+1):
                        f += ps[idx] * comb(o1, n) * x1**n * (1 - x1)**(o1 - n) * comb(o2[n], m) * x2**m * (1 - x2)**(o2[n] - m)
                        idx += 1            
            return f
    return f

def get_abcd_selection(h, axis_name, integrate_fail_x=True, integrate_pass_x=True, upper_bound=None):
    if type(h.axes[axis_name]) == hist.axis.Boolean:
        return 0, 1
    if axis_name not in abcd_thresholds:
        raise RuntimeError(f"Can not find threshold for abcd axis {axis_name}")
    ts = abcd_thresholds[axis_name]
    s = hist.tag.Slicer()
    if axis_name in ["iso", "dxy"]:
        # low is signal and high is sideband region
        lo = s[:complex(0,ts[1]):hist.sum] if integrate_pass_x else s[:complex(0,ts[1]):]
        if upper_bound is None:
            hi = s[complex(0,ts[1])::hist.sum] if integrate_fail_x else s[complex(0,ts[1])::] # no upper bound
        else:
            hi = s[complex(0,ts[1]):upper_bound:hist.sum] if integrate_fail_x else s[complex(0,ts[1]):upper_bound:] # upper bound
        return hi, lo
    elif axis_name in ["mt"]:
        # low is sideband and high is signal region
        lo = s[:complex(0,ts[-1]):hist.sum] if integrate_fail_x else s[:complex(0,ts[-1]):]
        hi = s[complex(0,ts[-1])::hist.sum] if integrate_pass_x else s[complex(0,ts[-1])::] # count overflow
        return lo, hi
    else:
        raise RuntimeError(f"No known abcd selection for {axis_name}")

def exp_fall_unc(x, a, b, c):
    return a * unp.exp(-b * x) + c

def exp_fall(x, a, b, c):
    return a * np.exp(-b * x) + c

def fit_multijet_bkg(x, y, y_err, x_eval=None, do_chi2=False):
    if x_eval is None:
        x_eval = x
    logger.info(f"Fit exp falling function --- ")

    p0=[sum(y)/0.15, 0.15, min(y)] # initial guesses
    params, cov = curve_fit(exp_fall, x, y, p0=p0, sigma=y_err, absolute_sigma=False)

    params = unc.correlated_values(params, cov)

    y_fit_u = exp_fall_unc(x_eval, *params)
    y_fit_err = np.array([y.s for y in y_fit_u])
    y_fit = np.array([y.n for y in y_fit_u])
    
    if do_chi2:
        y_pred_u = exp_fall_unc(x, *params)
        y_pred_err = np.array([y.s for y in y_pred_u])
        y_pred = np.array([y.n for y in y_pred_u])
        chisq = sum(((y - y_pred) / y_pred_err) ** 2) # use statistical error as expected from fit to avoid division through 0
        ndf = len(x)-len(params)
        logger.info(f"Fit result with chi2/ndf = {chisq:.2f}/{ndf} = {(chisq/ndf):.2f}")
        logger.info(f"Fit params={params}")

    return y_fit, y_fit_err

class HistselectorABCD(object):
    def __init__(self, h,
        integrate_fail_x=True, integrate_pass_x=True, integrate_fail_y=True, integrate_pass_y=True, 
        smooth_spectrum=False, 
        smoothing_axis_name="pt", 
        rebin_smoothing_axis="automatic", # can be a list of bin edges, "automatic", or None
        # rebin_smoothing_axis=[26,27,28,29,30,31,33,36,40,46,56], 
        upper_bound_y=None # using an upper bound on the abcd y-axis (e.g. isolation)
    ):
        self.integrate_fail_x = integrate_fail_x
        self.integrate_pass_x = integrate_pass_x
        self.integrate_fail_y = integrate_fail_y
        self.integrate_pass_y = integrate_pass_y

        self.upper_bound_y=upper_bound_y

        self.name_x = None
        self.name_y = None
        self.axis_x = None
        self.axis_y = None
        self.set_abcd_axes(h)

        self.fail_x=None
        self.pass_x=None
        self.fail_y=None
        self.pass_y=None
        self.set_abcd_selections(h)

        # perform a smoothing of the distribution before the abcd selection
        self.smooth_spectrum = smooth_spectrum
        self.smoothing_axis_name = smoothing_axis_name
        if rebin_smoothing_axis == "automatic":
            if smoothing_axis_name not in ["pt"]:
                raise RuntimeError(f"No automatic rebinning known for axis {smoothing_axis_name}")
            # try to find suitible binning
            edges = h.axes[smoothing_axis_name].edges.astype(int)
            target_edges = [26,27,28,29,30,31,33,36,40,46,56]
            # self.rebin_smoothing_axis = [x for x in sorted(set(edges).intersection(set(target_edges)))]
            self.rebin_smoothing_axis = [x for x in target_edges if x in edges]
            logger.info(f"For smoothing, axis {smoothing_axis_name} will be rebinned to {self.rebin_smoothing_axis}")
        else:
            self.rebin_smoothing_axis = rebin_smoothing_axis

        self.smoothing_axis_min = h.axes[smoothing_axis_name].edges[0] if self.rebin_smoothing_axis is None else self.rebin_smoothing_axis[0]
        self.smoothing_axis_max = h.axes[smoothing_axis_name].edges[-1] if self.rebin_smoothing_axis is None else self.rebin_smoothing_axis[-1]

    def set_abcd_axes(self, h):
        for a, b in abcd_variables:
            if self.name_x is None and a in h.axes.name: 
                self.name_x = a
                continue
            if self.name_x is None and b in h.axes.name: 
                self.name_x = b
                continue            
            if self.name_y is None and a in h.axes.name: 
                self.name_y = a
                continue
            if self.name_y is None and b in h.axes.name: 
                self.name_y = b
                continue
        if self.name_y is None:
            raise RuntimeError(f"Can not find ABCD axes among histogram axes {h.axes.name}")

        self.axis_x = h.axes[self.name_x]
        self.axis_y = h.axes[self.name_y]
        logger.info(f"Found ABCD axes {self.name_x} and {self.name_y}")

    def set_abcd_selections(self, h):
        self.fail_x, self.pass_x = get_abcd_selection(h, self.name_x, self.integrate_fail_x, self.integrate_pass_x)
        self.fail_y, self.pass_y = get_abcd_selection(h, self.name_y, self.integrate_fail_y, self.integrate_pass_y, upper_bound=self.upper_bound_y)

    def smoothen_spectrum(self, h, integration_axes=[], absval_axes=[], rebin_axes={}, flow=True):

        hNew = h.copy()
        if len(integration_axes) > 0:
            hNew = hNew[{n: slice(None,None,hist.sum) for n in integration_axes}]

        for ax in absval_axes:
            hNew = hh.makeAbsHist(hNew, ax, rename=False)

        if len(rebin_axes) > 0:
            hNew = hNew[{k:slice(None,None,hist.rebin(v)) for k,v in rebin_axes.items()}]

        oldshape = hNew.shape
        x_eval=None
        if self.rebin_smoothing_axis is not None:
            x_eval = h.axes[self.smoothing_axis_name].centers
            hNew = hh.rebinHist(hNew, self.smoothing_axis_name, self.rebin_smoothing_axis)

        axes = hNew.axes
        ax_idx = axes.name.index(self.smoothing_axis_name)
        values = hNew.values(flow=flow)
        variances = hNew.values(flow=flow)

        # put smoothing axis last
        if ax_idx != len(axes)-1:
            values = np.moveaxis(values, ax_idx, -1)
            variances = np.moveaxis(variances, ax_idx, -1)
        
        newshape = (np.product(values.shape[:-1]), values.shape[-1])
        values = np.reshape(values, newshape)
        variances = np.reshape(variances, newshape)

        # perform a fit in each bin
        x_widths = np.diff(axes[self.smoothing_axis_name].edges)
        x = axes[self.smoothing_axis_name].centers
        new_values = []
        new_variances = []
        for i in range(newshape[0]):
            logger.info(f"Now at {i}/{newshape[0]}")
            val, err = fit_multijet_bkg(x, values[i,:]/x_widths, np.sqrt(variances[i,:])/x_widths, x_eval=x_eval)
            # set the bin where the fit was performed 
            new_values.append(val)
            new_variances.append((err)**2)

        new_values = np.reshape(new_values, oldshape)
        new_variances = np.reshape(new_variances, oldshape)
        new_tensor = np.stack((new_values, new_variances), axis=-1)

        # bring back to original shape
        for k, v in rebin_axes.items():
            ax_idx = hNew.axes.name.index(k)
            new_tensor = np.repeat(new_tensor/v, repeats=v, axis=ax_idx)

        for ax in absval_axes:
            ax_idx = hNew.axes.name.index(ax)
            new_tensor = np.repeat(new_tensor/2., repeats=2, axis=ax_idx)

        for ax in integration_axes:
            ax_idx = h.axes.name.index(ax)
            ax_size = h.axes[ax].size
            # make new axis that was integrated out and broadcast to original shape 
            new_tensor = np.broadcast_to(new_tensor[...,np.newaxis], [*new_tensor.shape, h.axes[ax].size] )
            # move new axis to previous position
            new_tensor = np.moveaxis(new_tensor/ax_size, -1, ax_idx)

        newaxes = [np.newaxis if ax in integration_axes else None for ax in h.axes]
        h.view(flow=flow)[...] = new_tensor

        return h

class SignalSelectorABCD(HistselectorABCD):
    # simple ABCD method
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # signal region selection
    def get_hist(self, h):
        hSel = h[{self.name_x: self.pass_x, self.name_y: self.pass_y}]
        if self.smooth_spectrum:
            return self.smoothen_spectrum(hSel, integration_axes=["charge"], absval_axes=["eta"], rebin_axes={"eta": 2})
        return hSel

class FakeSelectorABCD(HistselectorABCD):
    # abstract base class for ABCD method for fakes
    def __init__(self, h, fakerate_axes=["eta","pt","charge"], *args, **kwargs):
        super().__init__(h, *args, **kwargs)
        self.fakerate_axes = fakerate_axes
        self.fakerate_integration_axes = [n for n in h.axes.name if n not in [self.name_x, self.name_y, *fakerate_axes]]
        logger.info(f"Setting fakerate integration axes to {self.fakerate_integration_axes}")

class FakeSelectorSimpleABCD(FakeSelectorABCD):
    # simple ABCD method
    def __init__(self, h, *args, **kwargs):
        super().__init__(h, *args, **kwargs)

    def get_hist(self, h):
        if any(a in h.axes.name for a in self.fakerate_integration_axes):
            fakerate_axes = [n for n in h.axes.name if n not in [*self.fakerate_integration_axes, self.name_x, self.name_y]]
            hA = h[{self.name_y: self.pass_y, self.name_x: self.fail_x}].project(*fakerate_axes)
            hB = h[{self.name_y: self.fail_y, self.name_x: self.fail_x}].project(*fakerate_axes)
        else:
            hA = h[{self.name_y: self.pass_y, self.name_x: self.fail_x}]
            hB = h[{self.name_y: self.fail_y, self.name_x: self.fail_x}]

        hFRF = hh.divideHists(hA, hB, cutoff=1, createNew=True)
        return hh.multiplyHists(hFRF, h[{self.name_y: self.fail_y, self.name_x: self.pass_x}])

class FakeSelectorSimultaneousABCD(FakeSelectorABCD):
    # simple ABCD method simultaneous fitting all regions
    def __init__(self, h, *args, **kwargs):
        super().__init__(h, *args, **kwargs)

    def get_hist(self, h):
        if h.storage_type == hist.storage.Weight:
            # setting errors to 0
            h.view(flow=True)[...] = np.stack((h.values(flow=True), np.zeros_like(h.values(flow=True))), axis=-1)

        if self.name_x not in h.axes.name or self.name_y not in h.axes.name:
            raise RuntimeError(f'{self.name_x} and {self.name_y} expected to be found in histogram, but only have axes {h.axes.name}')

        # axes in the correct ordering
        axes = [ax for ax in h.axes.name if ax not in [self.name_x, self.name_y]]
        axes += [self.name_x, self.name_y]

        if set(h.axes.name) != set(axes):
            logger.warning(f"Axes in histogram '{h.axes.name}' are not the same as required '{axes}' or in a different order than expected, try to project")
            h = h.project(*axes)

        # set the expected values in the signal region
        slices = [self.pass_x if n==self.name_x else self.pass_y if n==self.name_y else slice(None) for n in h.axes.name]
        h.values(flow=True)[*slices] = self.fakeHist(h).values(flow=True)
        return h

class FakeSelector1DExtendedABCD(FakeSelectorABCD):
    # extended ABCD method with 5 control regions as desribed in https://arxiv.org/abs/1906.10831 equation 16
    def __init__(self, h, *args, 
        smooth_fakerate=True, smoothing_order_fakerate=1, 
        # polynomial="power", 
        polynomial="bernstein", 
        **kwargs
    ):
        """
        :integrate_shapecorrection_x: if the shape correction factor for the abcd x-axis should be inclusive or differential
        :polynomial: choices: ["power","bernstein"]
        """
        super().__init__(h, *args, **kwargs)
        self.edges_x=None
        self.edges_y=None

        self.sel_x = None
        self.sel_dx = None
        self.sel_d2x = None
        self.sel_y = None
        self.sel_dy = None
        self.set_sidebands_selections_x()
        self.set_sidebands_selections_y()

        ### interpolate/smooth in x-axis in application region
        self.polynomial = polynomial 

        # fakerate factor
        self.smooth_fakerate = smooth_fakerate
        self.smoothing_order_fakerate = smoothing_order_fakerate

        # solve with non negative least squares
        if self.polynomial=="bernstein":
            self.solve = solve_nonnegative_leastsquare
        else:
            self.solve = solve_leastsquare

        # set smooth functions
        self.f_frf = None
        if self.smooth_fakerate:
            self.f_frf = get_smoothing_function_1D(self.smoothing_order_fakerate, pol=self.polynomial)

    # set slices object for selection of sideband regions
    def set_sidebands_selections_x(self):
        if self.name_x not in abcd_thresholds:
            raise RuntimeError(f"Can not find threshold for abcd axis {self.name_x}")
        ts = abcd_thresholds[self.name_x]
        s = hist.tag.Slicer()
        self.sel_dx = s[complex(0,ts[1]):complex(0,ts[2]):hist.sum]
        if self.name_x in ["mt"]:
            self.sel_x = s[complex(0,ts[2])::]
            self.sel_d2x = s[complex(0,ts[0]):complex(0,ts[1]):hist.sum]
        elif self.name_x in ["dxy", "iso"]:
            self.sel_x = s[complex(0,ts[0]):complex(0,ts[1]):hist.sum]
            self.sel_d2x = s[complex(0,ts[2]):complex(0,ts[3]):hist.sum]
        else:
            raise RuntimeError(f"Unknown thresholds for axis name {self.name_x}")

    def set_sidebands_selections_y(self):
        if self.name_y not in abcd_thresholds:
            raise RuntimeError(f"Can not find threshold for abcd axis {self.name_y}")
        ts = abcd_thresholds[self.name_y]
        s = hist.tag.Slicer()
        if self.name_y in ["mt"]:
            self.sel_y = s[complex(0,ts[2])::hist.sum]
            self.sel_dy = s[complex(0,ts[1]):complex(0,ts[2]):hist.sum]
        elif self.name_y in ["dxy", "iso"]:
            self.sel_y = s[complex(0,ts[0]):complex(0,ts[1]):hist.sum]
            if self.upper_bound_y is None:
                self.sel_dy = s[complex(0,ts[1])::hist.sum] # no upper bound
            else:
                self.sel_dy = s[complex(0,ts[1]):self.upper_bound_y:hist.sum] # upper bound            
        else:
            raise RuntimeError(f"Unknown thresholds for axis name {self.name_y}")

    def get_hist(self, h, h_nominal=None, variations_frf=False, flow=True):
        # TODO: take properly into account fakerate_integration_axes
        if variations_frf:
            storage = hist.storage.Double()
        else:
            storage = h.storage_type()
        idx_x = h.axes.name.index(self.name_x)
        if self.smooth_fakerate:

            if h_nominal is not None and id(h) != id(h_nominal):
                h = hh.transfer_variances(h, h_nominal)

            if len(self.fakerate_integration_axes) > 0:
                logger.info(f"Sum fakerate integration axes {self.fakerate_integration_axes}")
                idxs_i = [h.axes.name.index(n) for n in self.fakerate_integration_axes] # indices for broadcasting of axes that were integrated out
                hF = h[{n: hist.sum for n in self.fakerate_integration_axes}]
            else:
                idxs_i=[]
                hF = h

            y_frf, y_frf_var = self.compute_fakeratefactor(hF, syst_variations=variations_frf)
            c, cvar = self.get_yields_applicationregion(h)

            # broadcast abcd-x axis
            new_axis = [np.newaxis if i in [idx_x,*idxs_i] else slice(None) for i in range(c.ndim)]
            y_frf = y_frf[*new_axis] 
            y_frf_var = y_frf_var[*new_axis] if y_frf_var is not None else None

            d = c * y_frf

            if variations_frf:
                dvar = c[..., np.newaxis,np.newaxis] * y_frf_var[...,:,:]
                dvar = dvar - d[...,np.newaxis,np.newaxis]
            else:
                # only take bin by bin uncertainty from c region
                dvar = y_frf**2 * cvar
        else:
            # no smoothing of rates
            d, dvar = self.calculate_fullABCD(h)

        # set histogram in signal region
        if self.integrate_pass_x:
            d = d.sum(axis=idx_x)
            dvar = dvar.sum(axis=idx_x)

            axes = [a for a in h.axes if a.name not in [self.name_x, self.name_y]]
            hSignal = hist.Hist(*axes, storage=storage)
        else:
            hSignal = h[{self.name_x: self.sel_x, self.name_y: self.sel_y}]

        if variations_frf and self.smooth_fakerate:
            hSignal = self.get_syst_variations_hist(hSignal, d, dvar, flow=flow)
        else:
            hSignal.values(flow=flow)[...] = d
            if hSignal.storage_type == hist.storage.Weight:
                hSignal.variances(flow=flow)[...] = dvar

        return hSignal

    def get_syst_variations_hist(self, hNominal, values, variations, flow=True):
        # set systematic histogram with differences between variation and syst
        hsyst = hist.Hist(*hNominal.axes, hist.axis.Integer(0, variations.shape[-2], name="_param", overflow=True, underflow=True), common.down_up_axis, storage=hist.storage.Double())

        # check variations lower than 0
        vcheck = variations < (-1*values[...,np.newaxis,np.newaxis])
        if np.sum(vcheck) > 0:
            logger.warning(f"Found {np.sum(vcheck)} bins with variations giving negative yields, set these to 0")
            # variations[vcheck] = 0 

        hsyst.values(flow=flow)[...] = variations

        # decorrelate in fakerate axes
        axes_names = [n for n in self.fakerate_axes if n != self.smoothing_axis_name]
        hsyst = hh.expand_hist_by_duplicate_axes(hsyst, axes_names, [f"_{n}" for n in axes_names])    

        # add nominal hist and broadcast
        hNominal = hh.addHists(hNominal, hsyst)
        return hNominal

    def calculate_fullABCD(self, h, flow=True):
        if len(self.fakerate_integration_axes) > 0:
            raise NotImplementedError(f"Using fakerate integration axes is not supported in the binned extended ABCD method")

        ha = self.select_hist(h, selection={self.name_x: self.sel_dx, self.name_y: self.sel_dy})
        hax = self.select_hist(h, selection={self.name_x: self.sel_d2x, self.name_y: self.sel_dy})
        hb = self.select_hist(h, selection={self.name_x: self.sel_dx, self.name_y: self.sel_y})
        hbx = self.select_hist(h, selection={self.name_x: self.sel_d2x, self.name_y: self.sel_y})
        hc = self.select_hist(h, selection={self.name_x: self.sel_x, self.name_y: self.sel_dy})

        # calculate extended ABCD method without smoothing and in a single bin in x
        a = ha.values(flow=flow)
        ax = hax.values(flow=flow)
        b = hb.values(flow=flow)
        bx = hbx.values(flow=flow)
        c = hc.values(flow=flow)

        frf = (b/a)**2 * (ax/bx)
        frf = frf[...,np.newaxis]
        d = c * frf

        dvar=None
        if h.storage_type == hist.storage.Weight:
            avar = ha.variances(flow=flow)
            axvar = hax.variances(flow=flow)
            bvar = hb.variances(flow=flow)
            bxvar = hbx.variances(flow=flow)
            cvar = hc.variances(flow=flow)
            dvar = frf**2 * cvar + d**2 * (4 * bvar/b**2 + 4 * avar/a**2 + axvar/ax**2 + bxvar/bx**2)[...,np.newaxis]
        
        return d, dvar

    def select_hist(self, h, selection={}):
        hSel = h[selection]
        if self.smooth_spectrum:
            hSel = self.smoothen_spectrum(hSel, integration_axes=["charge"], absval_axes=["eta"], rebin_axes={"eta": 2})
        return hSel

    def get_yields_applicationregion(self, h, flow=True):
        hC = h[{self.name_x: self.sel_x}]
        c = hC[{self.name_y: self.sel_dy}].values(flow=flow)
        if h.storage_type == hist.storage.Weight:
            cvar = hC[{self.name_y: self.sel_dy}].variances(flow=flow)
            return c, cvar
        return c, None

    def compute_fakeratefactor(self, h, syst_variations=False, flow=True, auxiliary_info=False):

        # rebin in smoothing axis to have stable ratios
        hNew = hh.rebinHist(h, self.smoothing_axis_name, self.rebin_smoothing_axis) if self.rebin_smoothing_axis is not None else h

        # select sideband regions
        a = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].values(flow=flow)
        ax = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].values(flow=flow)
        b = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].values(flow=flow)
        bx = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].values(flow=flow)

        if h.storage_type == hist.storage.Weight:
            avar = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].variances(flow=flow)
            axvar = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].variances(flow=flow)
            bvar = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].variances(flow=flow)
            bxvar = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].variances(flow=flow)

        # fakerate factor
        y_num = ax*b**2
        y_den = bx*a**2
        y = y_num/y_den

        if (y_den <= 0).sum() > 0:
            logger.warning(f"{(y_den < 0).sum()} bins with negative and {(y_den == 0).sum()} bins with empty content found for denominator in the fakerate factor.")
        
        if h.storage_type == hist.storage.Weight:
            # full variances
            y_var = y**2 * (axvar/ax**2 + 4*bvar/b**2 + 4*avar/a**2 + bxvar/bx**2)

        if self.smooth_fakerate:
            x = self.get_bin_centers_smoothing(hNew, flow=False) # the bins where the smoothing is performed (can be different to the bin in h)
            return self.smoothen_fakerate(h, x, y, y_var, syst_variations=syst_variations, auxiliary_info=auxiliary_info, flow=flow)
        else:
            return y, y_var

    def smoothen_fakerate(self, h, x, y, y_var, syst_variations=False, auxiliary_info=False, flow=True):
        if h.storage_type == hist.storage.Weight:
            # transform with weights
            w = 1/np.sqrt(y_var)
        else:
            logger.warning("Smoothing extended ABCD on histogram without uncertainties, make an unweighted linear squared solution.")
            w = np.ones_like(y)

        # move smoothing axis to last
        axes = [n for n in h.axes.name if n not in [self.name_x, self.name_y] ]
        idx_ax_smoothing = axes.index(self.smoothing_axis_name)
        if idx_ax_smoothing != len(axes)-1:
            y = np.moveaxis(y, idx_ax_smoothing, -1)
            w = np.moveaxis(w, idx_ax_smoothing, -1)

        # smooth frf (e.g. in pT)
        X, XTY = get_parameter_matrices(x, y, w, self.smoothing_order_fakerate, pol=self.polynomial)
        
        params, cov = self.solve(X, XTY)

        # evaluate in range of original histogram
        x_smooth_orig = self.get_bin_centers_smoothing(h, flow=False)
        y_smooth_orig = self.f_frf(x_smooth_orig, params)

        if syst_variations:
            y_smooth_var_orig = self.make_eigen_variations_frf(params, cov, x_smooth_orig)
        else: 
            y_smooth_var_orig = None

        # move smoothing axis to original positon again
        if idx_ax_smoothing != len(axes)-1:
            y_smooth_orig = np.moveaxis(y_smooth_orig, -1, idx_ax_smoothing)
            y_smooth_var_orig = np.moveaxis(y_smooth_var_orig, -3, idx_ax_smoothing) if syst_variations else None

        # check for negative rates
        if np.sum(y_smooth_orig<0) > 0:
            logger.warning(f"Found {np.sum(y_smooth_orig<0)} bins with negative fake rate factors")
        if y_smooth_var_orig is not None and np.sum(y_smooth_var_orig<0) > 0:
            logger.warning(f"Found {np.sum(y_smooth_var_orig<0)} bins with negative fake rate factor variations")

        if auxiliary_info:
            y_pred = self.f_frf(x, params)
            # flatten
            y_pred = y_pred.reshape(y.shape) 
            w = w.reshape(y.shape)
            chi2, ndf = compute_chi2(y, y_pred, w, nparams=params.shape[-1])
            return y_smooth_orig, y_smooth_var_orig, params, cov, chi2, ndf
        else:
            return y_smooth_orig, y_smooth_var_orig

    def make_eigen_variations_frf(self, params, cov, x1, x2=None):
        return make_eigen_variations(params, cov, func=self.f_frf, x1=x1, x2=x2, force_positive=self.polynomial=="bernstein")

    def get_bin_centers_smoothing(self, h, flow=False):
        return self.get_bin_centers(h, self.smoothing_axis_name, self.smoothing_axis_min, self.smoothing_axis_max, flow=flow)

    def get_bin_centers(self, h, axis_name, xmin=None, xmax=None, flow=True, cap=False):
        # get bin centers for interpolation/smoothing
        x = h.axes[axis_name].centers
        if flow:
            # extend array for underflow/overflow with distance from difference of two closest values 
            if h.axes[axis_name].traits.underflow:
                new_x = x[0] - x[1] + x[0]
                x = np.array([new_x, *x])
                logger.debug(f"Extend bin center array by underflow bin using {new_x}")
            if h.axes[axis_name].traits.overflow:
                new_x = x[-1] + x[-1] - x[-2]
                x = np.array([*x, new_x])
                logger.debug(f"Extend bin center array by overflow bin using {new_x}")
        if self.polynomial=="bernstein":
            # transform bernstein polinomials to [0,1]
            x = (x - xmin) / (xmax - xmin)
            if np.sum(x < 0) or np.sum(x > 1):
                if cap:
                    logger.info("Values outside [0,1] found, those will be capped to [0,1]")
                    x[x < 0] = 0
                    x[x > 1] = 1
                else:
                    raise RuntimeError(f"All values need to be within [0,1] but {np.sum(x < 0)} values smaller 0 and {np.sum(x > 1)} larger 1 found after transformation with xmin={xmin} and xmax={xmax}")
        return x

class FakeSelector2DExtendedABCD(FakeSelector1DExtendedABCD):
    # extended ABCD method with 8 control regions as desribed in https://arxiv.org/abs/1906.10831 equation 15
    def __init__(self, h, *args, 
        interpolate_x=True, interpolation_order=2, #rebin_x=[0,20,40,45,50,60],
        rebin_x="automatic", # can be a list of bin edges, "automatic", or None
        #rebin_x=[0, 20, 40, 44, 49, 55, 62],
        integrate_shapecorrection_x=False, smooth_shapecorrection=True, smoothing_order_shapecorrection=[2,2,2],
        **kwargs
    ):
        super().__init__(h, *args, **kwargs)

        self.sel_d2y = None
        self.set_sidebands_selections_y()

        self.interpolate_x = interpolate_x
        self.interpolation_order = interpolation_order

        # rebin x for interpolation but evaluate at original binning
        if rebin_x == "automatic":
            if self.name_x not in ["mt"]:
                raise RuntimeError(f"No automatic rebinning known for axis {self.name_x}")
            # try to find suitible binning
            edges = h.axes[self.name_x].edges.astype(int)
            target_edges = [0, 20, 40, 44, 49, 55, 62]
            self.rebin_x = [x for x in target_edges if x in edges]
            logger.info(f"For interpolation, axis {self.name_x} will be rebinned to {self.rebin_x}")
        else:
            self.rebin_x = rebin_x

        # min and max (in application region) for transformation of bernstain polynomials into interval [0,1]
        self.axis_x_min = h[{self.name_x: self.sel_x}].axes[self.name_x].edges[0]
        if self.name_x =="mt":
            # mt does not have an upper bound, cap at 80
            self.axis_x_max = 80
        elif self.name_x in ["iso", "dxy"]:
            # iso and dxy have a finite lower and upper bound in the application region
            self.axis_x_max = abcd_thresholds[self.name_x][1]

        # shape correction, can be interpolated in the abcd x-axis in 1D, in the x-axis and smoothing axis in 2D, or in the smoothing axis integrating out the abcd x-axis in 1D
        self.integrate_shapecorrection_x = integrate_shapecorrection_x # if the shape correction factor for the abcd x-axis should be inclusive or differential
        if self.integrate_shapecorrection_x and self.interpolate_x:
            raise RuntimeError("Can not integrate and interpolate x at the same time")
        self.smooth_shapecorrection = smooth_shapecorrection
        self.smoothing_order_shapecorrection = smoothing_order_shapecorrection

        # set smooth functions
        self.f_scf = None
        if self.interpolate_x and self.smooth_shapecorrection: 
            self.f_scf = get_smoothing_function_2D(self.interpolation_order, self.smoothing_order_shapecorrection, pol=self.polynomial)
        elif self.interpolate_x:
            self.f_scf = get_smoothing_function_1D(self.interpolation_order, pol=self.polynomial)
        elif self.smooth_shapecorrection:
            self.f_scf = get_smoothing_function_1D(self.smoothing_order_shapecorrection, pol=self.polynomial)            

    # set slices object for selection of sideband regions
    def set_sidebands_selections_y(self):
        if self.name_y not in abcd_thresholds:
            raise RuntimeError(f"Can not find threshold for abcd axis {self.name_y}")
        ts = abcd_thresholds[self.name_y]
        s = hist.tag.Slicer()
        self.sel_dy = s[complex(0,ts[1]):complex(0,ts[2]):hist.sum]
        if self.name_y in ["mt"]:
            self.sel_y = s[complex(0,ts[2])::hist.sum]
            self.sel_d2y = s[complex(0,ts[0]):complex(0,ts[1]):hist.sum]
        elif self.name_y in ["dxy", "iso"]:
            self.sel_y = s[complex(0,ts[0]):complex(0,ts[1]):hist.sum]
            if self.upper_bound_y is None:
                self.sel_d2y = s[complex(0,ts[2])::hist.sum] # no upper bound
            else:
                self.sel_d2y = s[complex(0,ts[2]):hist.overflow:hist.sum] # upper bound, no overflow

    def get_hist(self, h, h_nominal=None, variations_scf=False, variations_frf=False, flow=True):
        if variations_scf and variations_frf:
            raise RuntimeError(f"Can only calculate vairances for fakerate factor or shape correction factor but not both")
        elif variations_scf or variations_frf:
            storage = hist.storage.Double()
        else:
            storage = h.storage_type()

        idx_x = h.axes.name.index(self.name_x) # indices for broadcasting of x-axes used in ABCD
        if self.smooth_fakerate or self.interpolate_x or self.smooth_shapecorrection:
            if h_nominal is not None and id(h) != id(h_nominal):
                h = hh.transfer_variances(h, h_nominal)
            
            if len(self.fakerate_integration_axes) > 0:
                logger.info(f"Sum fakerate integration axes {self.fakerate_integration_axes}")
                idxs_i = [h.axes.name.index(n) for n in self.fakerate_integration_axes] # indices for broadcasting of axes that were integrated out
                hF = h[{n: hist.sum for n in self.fakerate_integration_axes}]
            else:
                idxs_i=[]
                hF = h
                
            y_frf, y_frf_var = self.compute_fakeratefactor(hF, syst_variations=variations_frf)
            y_scf, y_scf_var = self.compute_shapecorrection(hF, syst_variations=variations_scf)
            c, cy, cvar, cyvar = self.get_yields_applicationregion(h)

            # broadcast abcd-x axis
            new_axis = [np.newaxis if i in [idx_x,*idxs_i] else slice(None) for i in range(c.ndim)]
            y_frf = y_frf[*new_axis] 
            y_frf_var = y_frf_var[*new_axis] if y_frf_var is not None else None
            if self.integrate_shapecorrection_x:
                y_scf = y_scf[*new_axis]
                y_scf_var = y_scf_var[*new_axis] if y_scf_var is not None else None
            else:
                new_axis = [np.newaxis if i in idxs_i else slice(None) for i in range(c.ndim)]
                y_scf = y_scf[*new_axis]
                y_scf_var = y_scf_var[*new_axis] if y_scf_var is not None else None

            d = c * y_scf * y_frf

            if variations_scf and (self.interpolate_x or self.smooth_shapecorrection):
                dvar = c[..., np.newaxis,np.newaxis] * y_frf[...,np.newaxis,np.newaxis] * y_scf_var[...,:,:]
                dvar = dvar - d[...,np.newaxis,np.newaxis]
            elif variations_frf and self.smooth_fakerate:
                dvar = c[..., np.newaxis,np.newaxis] * y_scf[..., np.newaxis,np.newaxis] * y_frf_var[...,:,:]
                dvar = dvar - d[...,np.newaxis,np.newaxis]
            elif self.smooth_shapecorrection or self.interpolate_x:
                # only take bin by bin uncertainty from c region
                dvar = (y_scf * y_frf)**2 * cvar
            else:
                # # take bin by bin uncertainty from c * c/cy 
                dvar = y_frf**2 *( 4*(c/cy)**2 * cvar + (c/cy**2)**2 * cyvar )
        else:
            # no smoothing of rates
            d, dvar = self.calculate_fullABCD(h)

        # set histogram in signal region
        if self.integrate_pass_x:
            d = d.sum(axis=idx_x)
            dvar = dvar.sum(axis=idx_x)

            axes = [a for a in h.axes if a.name not in [self.name_x, self.name_y]]
            hSignal = hist.Hist(*axes, storage=storage)
        else:
            hSignal = h[{self.name_x: self.sel_x, self.name_y: self.sel_y}] 

        if (variations_scf and (self.interpolate_x or self.smooth_shapecorrection)) or (variations_frf and self.smooth_fakerate):
            hSignal = self.get_syst_variations_hist(hSignal, d, dvar, flow=flow)
        else:
            hSignal.values(flow=flow)[...] = d
            if hSignal.storage_type == hist.storage.Weight:
                hSignal.variances(flow=flow)[...] = dvar

        return hSignal

    def compute_shapecorrection(self, h, syst_variations=False, flow=True, auxiliary_info=False):
        # rebin in smoothing axis to have stable ratios
        hNew = hh.rebinHist(h, self.smoothing_axis_name, self.rebin_smoothing_axis) if self.rebin_smoothing_axis is not None else h
        hNew = hh.rebinHist(hNew, self.name_x, self.rebin_x) if self.rebin_x is not None else hNew 
        hNew = hNew[{self.name_x: self.sel_x}]

        c = hNew[{self.name_y: self.sel_dy}].values(flow=flow)
        cy = hNew[{self.name_y: self.sel_d2y}].values(flow=flow)
        if h.storage_type == hist.storage.Weight:
            cvar = hNew[{self.name_y: self.sel_dy}].variances(flow=flow)
            cyvar = hNew[{self.name_y: self.sel_d2y}].variances(flow=flow)

        if self.integrate_shapecorrection_x:
            idx_x = hNew.axes.name.index(self.name_x)
            c = c.sum(axis=idx_x)
            cy = cy.sum(axis=idx_x)
            cvar = cvar.sum(axis=idx_x)
            cyvar = cyvar.sum(axis=idx_x)

        # shape correction factor
        y = c/cy
        if (cy <= 0).sum() > 0:
            logger.warning(f"{(cy < 0).sum()} bins with negative and {(cy == 0).sum()} bins with empty content found for denominator in the shape correction factor.")
        
        if h.storage_type == hist.storage.Weight:
            # y_var = y**2 * (cvar/c**2 + cyvar/cy**2 )
            y_var = cvar/cy**2 + (c**2 * cyvar)/cy**4 # multiply out for better numerical stability (avoid NaNs from divisions)

        if self.interpolate_x or self.smooth_shapecorrection:
            if h.storage_type == hist.storage.Weight:
                w = 1/np.sqrt(y_var)
            else:
                logger.warning("Smoothing extended ABCD on histogram without uncertainties, make an unweighted linear squared solution.")
                w = np.ones_like(y)

        if self.interpolate_x:
            axes = [n for n in h.axes.name if n not in [self.name_y] ]

            idx_ax_interpol = axes.index(self.name_x)

            # constract x points in application region
            x_interpol = self.get_bin_centers_interpolation(hNew, flow=flow)
            # x axis in original binning in passing region
            x_interpol_orig = self.get_bin_centers_interpolation(h[{self.name_x: self.sel_x}], flow=flow, cap=True)

            if self.smooth_shapecorrection:
                # interpolate scf in mT and smooth in pT (i.e. 2D)

                idx_ax_smoothing = hNew.axes.name.index(self.smoothing_axis_name)
                if idx_ax_smoothing != len(axes)-2 or idx_ax_interpol != len(axes)-1:
                    y = np.moveaxis(y, (idx_ax_smoothing, idx_ax_interpol), (-2, -1))
                    w = np.moveaxis(w, (idx_ax_smoothing, idx_ax_interpol), (-2, -1))

                x_smoothing = self.get_bin_centers_smoothing(hNew, flow=False)

                X, y, XTY = get_parameter_matrices_from2D(x_interpol, x_smoothing, y, w, 
                    self.interpolation_order, self.smoothing_order_shapecorrection, pol=self.polynomial, flatten=True)

                params, cov = self.solve(X, XTY)

                x_smooth_orig = self.get_bin_centers_smoothing(h, flow=False)
                y_smooth_orig = self.f_scf(x_interpol_orig, x_smooth_orig, params)

                if syst_variations:
                    y_smooth_var_orig = self.make_eigen_variations_scf(params, cov, x_interpol_orig, x_smooth_orig)
                else: 
                    y_smooth_var_orig = None

                # move interpolation axis to original positon again
                if idx_ax_smoothing != len(axes)-2 or idx_ax_interpol != len(axes)-1:
                    y_smooth_orig = np.moveaxis(y_smooth_orig, (-2, -1), (idx_ax_smoothing, idx_ax_interpol))
                    y_smooth_var_orig = np.moveaxis(y_smooth_var_orig, (-4, -3), (idx_ax_smoothing, idx_ax_interpol)) if syst_variations else None

                if auxiliary_info:
                    y_pred = self.f_scf(x_interpol, x_smoothing, params)
            else:
                # interpolate scf in mT in 1D
                if idx_ax_interpol != len(axes)-1:
                    y = np.moveaxis(y, idx_ax_interpol, -1)
                    w = np.moveaxis(w, idx_ax_interpol, -1)

                X, XTY = get_parameter_matrices(x_interpol, y, w, self.smoothing_order_fakerate, pol=self.polynomial)
                params, cov = self.solve(X, XTY)

                y_smooth_orig = self.f_scf(x_interpol_orig, params)

                if syst_variations:
                    y_smooth_var_orig = self.make_eigen_variations_scf(params, cov, x_interpol_orig)
                else: 
                    y_smooth_var_orig = None

                if auxiliary_info:
                    y_pred = self.f_scf(x_interpol, params)

                # move interpolation axis to original positon again
                if idx_ax_interpol != len(axes)-1:
                    y_smooth_orig = np.moveaxis(y_smooth_orig, -1, idx_ax_interpol)
                    y_smooth_var_orig = np.moveaxis(y_smooth_var_orig, -3, idx_ax_smoothing) if syst_variations else None
        
            # check for negative rates
            if np.sum(y_smooth_orig<0) > 0:
                logger.warning(f"Found {np.sum(y_smooth_orig<0)} bins with negative shape correction factors")
            if y_smooth_var_orig is not None and np.sum(y_smooth_var_orig<0) > 0:
                logger.warning(f"Found {np.sum(y_smooth_var_orig<0)} bins with negative shape correction factor variations")

            if auxiliary_info:
                # flatten
                y_pred = y_pred.reshape(y.shape) 
                w = w.reshape(y.shape)
                chi2, ndf = compute_chi2(y, y_pred, w, nparams=params.shape[-1])
                return y_smooth_orig, y_smooth_var_orig, params, cov, chi2, ndf
            else:
                return y_smooth_orig, y_smooth_var_orig

        elif self.smooth_shapecorrection:
            # don't interpolate in mT, but smooth in pT in 1D

            # move smoothing axis to last
            axes = [n for n in h.axes.name if n not in [self.name_x, self.name_y] ]
            idx_ax_smoothing = axes.index(self.smoothing_axis_name)
            if idx_ax_smoothing != len(axes)-1:
                y = np.moveaxis(y, idx_ax_smoothing, -1)
                w = np.moveaxis(w, idx_ax_smoothing, -1)

            # smooth scf (e.g. in pT)
            x_smoothing = self.get_bin_centers_smoothing(hNew, flow=False)
            X, XTY = get_parameter_matrices(x_smoothing, y, w, self.smoothing_order_shapecorrection, pol=self.polynomial)
            params, cov = self.solve(X, XTY)

            # evaluate in range of original histogram
            x_smooth_orig = self.get_bin_centers_smoothing(h, flow=False)
            y_smooth_orig = self.f_scf(x_smooth_orig, params)

            if syst_variations:
                y_smooth_var_orig = self.make_eigen_variations_scf(params, cov, x_smooth_orig)
            else: 
                y_smooth_var_orig = None

            # move smoothing axis to original positon again
            if idx_ax_smoothing != len(axes)-1:
                y_smooth_orig = np.moveaxis(y_smooth_orig, -1, idx_ax_smoothing)
                y_smooth_var_orig = np.moveaxis(y_smooth_var_orig, -3, idx_ax_smoothing) if syst_variations else None

            # check for negative rates
            if np.sum(y_smooth_orig<0) > 0:
                logger.warning(f"Found {np.sum(y_smooth_orig<0)} bins with negative shape correction factors")
            if y_smooth_var_orig is not None and np.sum(y_smooth_var_orig<0) > 0:
                logger.warning(f"Found {np.sum(y_smooth_var_orig<0)} bins with negative shape correction factor variations")

            if auxiliary_info:
                y_pred = self.f_scf(x_smoothing, params)
                # flatten
                y_pred = y_pred.reshape(y.shape) 
                w = w.reshape(y.shape)
                chi2, ndf = compute_chi2(y, y_pred, w, nparams=params.shape[-1])
                return y_smooth_orig, y_smooth_var_orig, params, cov, chi2, ndf
            else:
                return y_smooth_orig, y_smooth_var_orig
        else:
            return y, y_var

    def compute_fakeratefactor(self, h, syst_variations=False, flow=True, throw_toys=False, auxiliary_info=False):

        # rebin in smoothing axis to have stable ratios
        hNew = hh.rebinHist(h, self.smoothing_axis_name, self.rebin_smoothing_axis) if self.rebin_smoothing_axis is not None else h

        # select sideband regions
        a = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].values(flow=flow)
        ax = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].values(flow=flow)
        ay = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_d2y}].values(flow=flow)
        axy = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_d2y}].values(flow=flow)
        b = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].values(flow=flow)
        bx = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].values(flow=flow)

        if h.storage_type == hist.storage.Weight:
            avar = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].variances(flow=flow)
            axvar = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].variances(flow=flow)
            ayvar = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_d2y}].variances(flow=flow)
            axyvar = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_d2y}].variances(flow=flow)
            bvar = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].variances(flow=flow)
            bxvar = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].variances(flow=flow)

        # fakerate factor
        y_num = (ax*ay*b)**2
        y_den = a**4 * axy * bx
        y = y_num/y_den

        if throw_toys:
            logger.info("Throw toys")
            # throw toys for each parameter separately
            values = np.stack([a, ax, ay, axy, b, bx], axis=-1)
            variances = np.stack([avar, axvar, ayvar, axyvar, bvar, bxvar], axis=-1)
            
            # # throw toys for nominator and denominator instead
            # y_num_var = y_num**2 * (4*axvar/ax**2 + 4*ayvar/ay**2 + 4*bvar/b**2)
            # y_den_var = y_den**2 * (16*avar/a**2 + axyvar/axy**2 + bxvar/bx**2)
            # values = np.stack([y_num, y_den], axis=-1)
            # variances = np.stack([y_num_var, y_den_var], axis=-1)

            np.random.seed(42)  # For reproducibility
            nsamples=10000
            toy_shape = [*values.shape, nsamples]
            toy_size=np.product(toy_shape)
            # throw gaussian toys
            toys = np.random.normal(0, 1, size=toy_size)
            toys = np.reshape(toys, toy_shape)
            toys = toys*np.sqrt(variances)[...,np.newaxis] + values[...,np.newaxis]

            toy = (toys[...,1,:]*toys[...,2,:]*toys[...,4,:])**2 / (toys[...,0,:]**4 * toys[...,3,:] * toys[...,5,:])
            # toy = toys[...,0,:] / toys[...,1,:]

            toy_mean = np.mean(toy, axis=-1)
            toy_var = np.var(toy, ddof=1, axis=-1) 

            y = toy_mean
            y_var = toy_var

            logger.info("Done with toys")

        if (y_den <= 0).sum() > 0:
            logger.warning(f"{(y_den < 0).sum()} bins with negative and {(y_den == 0).sum()} bins with empty content found for denominator in the fakerate factor.")
        
        if h.storage_type == hist.storage.Weight:
            # full variances
            y_var = y**2 * (4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 16*avar/a**2 + bxvar/bx**2)

        if self.smooth_fakerate:
            x = self.get_bin_centers_smoothing(hNew, flow=False) # the bins where the smoothing is performed (can be different to the bin in h)
            return self.smoothen_fakerate(h, x, y, y_var, syst_variations=syst_variations, auxiliary_info=auxiliary_info)
        else:
            return y, y_var

    def make_eigen_variations_scf(self, params, cov, x1, x2=None):
        return make_eigen_variations(params, cov, func=self.f_scf, x1=x1, x2=x2, force_positive=self.polynomial=="bernstein")

    def get_bin_centers_interpolation(self, h, flow=True, cap=False):
        return self.get_bin_centers(h, self.name_x, self.axis_x_min, self.axis_x_max, flow=flow, cap=cap)

    def get_yields_applicationregion(self, h, flow=True):
        hC = h[{self.name_x: self.sel_x}]
        c = hC[{self.name_y: self.sel_dy}].values(flow=flow)
        cy = hC[{self.name_y: self.sel_d2y}].values(flow=flow)
        if h.storage_type == hist.storage.Weight:
            cvar = hC[{self.name_y: self.sel_dy}].variances(flow=flow)
            cyvar = hC[{self.name_y: self.sel_d2y}].variances(flow=flow)
            return c, cy, cvar, cyvar
        return c, cy, None, None

    def calculate_fullABCD(self, h, flow=True):
        if len(self.fakerate_integration_axes) > 0:
            raise NotImplementedError(f"Using fakerate integration axes is not supported in the binned extended ABCD method")

        ha = self.select_hist(h, selection={self.name_x: self.sel_dx, self.name_y: self.sel_dy})
        hax = self.select_hist(h, selection={self.name_x: self.sel_d2x, self.name_y: self.sel_dy})
        hay = self.select_hist(h, selection={self.name_x: self.sel_dx, self.name_y: self.sel_d2y})
        haxy = self.select_hist(h, selection={self.name_x: self.sel_d2x, self.name_y: self.sel_d2y})
        hb = self.select_hist(h, selection={self.name_x: self.sel_dx, self.name_y: self.sel_y})
        hbx = self.select_hist(h, selection={self.name_x: self.sel_d2x, self.name_y: self.sel_y})
        hc = self.select_hist(h, selection={self.name_x: self.sel_x, self.name_y: self.sel_dy})
        hcy = self.select_hist(h, selection={self.name_x: self.sel_x, self.name_y: self.sel_d2y})

        # calculate extended ABCD method without smoothing and in a single bin in x
        a = ha.values(flow=flow)
        ax = hax.values(flow=flow)
        ay = hay.values(flow=flow)
        axy = haxy.values(flow=flow)
        b = hb.values(flow=flow)
        bx = hbx.values(flow=flow)
        c = hc.values(flow=flow)
        cy = hcy.values(flow=flow)

        d = c * (c*(ax*ay*b)[...,np.newaxis])**2 / ((a**4 * axy*bx)[...,np.newaxis]*cy)

        dvar=None
        if h.storage_type == hist.storage.Weight:
            avar = ha.variances(flow=flow)
            axvar = hax.variances(flow=flow)
            ayvar = hay.variances(flow=flow)
            axyvar = haxy.variances(flow=flow)
            bvar = hb.variances(flow=flow)
            bxvar = hbx.variances(flow=flow)
            cvar = hc.variances(flow=flow)
            cyvar = hcy.variances(flow=flow)
            dvar = d**2 * ((4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 16*avar/a**2 + bxvar/bx**2)[...,np.newaxis] + 4*cvar/c**2 + cyvar/cy**2 )

        return d, dvar
