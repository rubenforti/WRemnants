import hist
import numpy as np
from utilities import boostHistHelpers as hh
from utilities import common, logging
from utilities.fnnls import fnnls
from scipy.optimize import nnls
from scipy.special import comb
import pdb

logger = logging.child_logger(__name__)

abcd_thresholds={
    "mt":40,
    "iso":0.15,
    "dxy":0.01
}
# default abcd_variables to look for
abcd_variables = (("mt", "passMT"), ("iso", "passIso"), ("dxy", "passDxy"))

def get_eigen_variations(cov, params):
    # diagonalize and get eigenvalues and eigenvectors
    e, v = np.linalg.eigh(cov)            
    magT = np.transpose(np.sqrt(e)[...,np.newaxis]*v, axes=(*np.arange(params.ndim-1), -1, -2))
    params_var = np.transpose(params[...,np.newaxis] + magT, axes=(*np.arange(params.ndim-1), -1, -2))
    return params_var

def get_parameter_matrices(x, y, w, order, pol="power", mask=None):
    # TODO: Mask if needed
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

def get_parameter_matrices_from2D(x, x2, y, w, order, order2=None, pol="power", mask=None, flatten=False):
    # TODO: Mask if needed
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

def get_abcd_selection(h, axis_name, integrate_fail_x=True, integrate_pass_x=True):
    if type(h.axes[axis_name]) == hist.axis.Boolean:
        return 0, 1
    if axis_name not in abcd_thresholds:
        raise RuntimeError(f"Can not find threshold for abcd axis {axis_name}")
    tBin = h.axes[axis_name].index(abcd_thresholds[axis_name])
    s = hist.tag.Slicer()
    if axis_name in ["mt",]:
        lo = s[:tBin:hist.sum] if integrate_fail_x else s[:tBin:]
        hi = s[tBin::hist.sum] if integrate_pass_x else s[tBin::]
        return lo, hi
    else:
        lo = s[:tBin:hist.sum] if integrate_pass_x else s[:tBin:]
        hi = s[tBin::hist.sum] if integrate_fail_x else s[tBin::]
        return hi, lo


class HistselectorABCD(object):
    def __init__(self, h,
        integrate_fail_x=True, integrate_pass_x=True, integrate_fail_y=True, integrate_pass_y=True, 
        smooth_spectrum=False, smoothing_spectrum_name="pt"
    ):
        self.integrate_fail_x = integrate_fail_x
        self.integrate_pass_x = integrate_pass_x
        self.integrate_fail_y = integrate_fail_y
        self.integrate_pass_y = integrate_pass_y

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
        self.smoothing_spectrum_name = smoothing_spectrum_name

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
        else:
            self.axis_x = h.axes[self.name_x]
            self.axis_y = h.axes[self.name_y]
            logger.info(f"Found ABCD axes {self.name_x} and {self.name_y}")

    def set_abcd_selections(self, h):
        self.fail_x, self.pass_x = get_abcd_selection(h, self.name_x, self.integrate_fail_x, self.integrate_pass_x)
        self.fail_y, self.pass_y = get_abcd_selection(h, self.name_y, self.integrate_fail_y, self.integrate_pass_y)

class SignalSelectorABCD(HistselectorABCD):
    # simple ABCD method
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # signal region selection
    def get_hist(self, h):
        return h[{self.name_x: self.pass_x, self.name_y: self.pass_y}]

class FakeSelectorABCD(HistselectorABCD):
    # simple ABCD method
    def __init__(self, h, fakerate_axes, *args, **kwargs):
        super().__init__(h, *args, **kwargs)
        self.fakerate_axes = fakerate_axes
        self.fakerate_integration_axes = [n for n in h.axes.name if n not in [self.name_x, self.name_y, *fakerate_axes]]
        logger.info(f"Setting fakerate integration axes to {self.fakerate_integration_axes}")

class FakeSelectorSimpleABCD(FakeSelectorABCD):
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

class FakeSelectorExtendedABCD(FakeSelectorABCD):
    # extended ABCD method as desribed in https://arxiv.org/abs/1906.10831 equation 15
    def __init__(self, h, *args, **kwargs):
        super().__init__(h, *args, **kwargs)
        self.edges_x=None
        self.edges_y=None

        self.sel_x = None
        self.sel_dx = None
        self.sel_d2x = None
        self.sel_y = None
        self.sel_dy = None
        self.sel_d2y = None
        self.set_sidebands_selections()

        self.rebin_pt = [26,27,28,29,30,31,33,36,40,46,56]
        # self.rebin_pt = [26,31,36,41,46,56]
        self.rebin_x = None if len(self.axis_x.edges) < 10 else [0,20,40,44,49,55,62]

        # nominal histograms to transfer variances to systematics
        self.use_container = True
        if self.use_container:
            hnew = hh.rebinHist(h, "pt", self.rebin_pt) if self.rebin_pt is not None else h
            self.h_fakerate = hnew
            self.h_shapecorrection = hnew[{self.name_x: self.sel_x}]

        ### interpolate/smooth in x-axis
        self.smoothing_axis_name="pt"
        self.polynomial = "bernstein" # alternaive "power" for power series
        self.smoothing_axis_max = self.rebin_pt[0]
        self.smoothing_axis_min = self.rebin_pt[-1]

        edges_x = h[{self.name_x: self.sel_x}].axes[self.name_x].edges
        if h[{self.name_x: self.sel_x}].axes[self.name_x].traits.underflow:
            edges_x = np.array([edges_x[0]-np.diff(edges_x[:2]), *edges_x])
        if h[{self.name_x: self.sel_x}].axes[self.name_x].traits.overflow:
            edges_x = np.append(edges_x, edges_x[-1]+np.diff(edges_x[-2:]))
        self.axis_x_min = edges_x[0]
        self.axis_x_max = edges_x[-1]

        # shape correction, can be interpolated in the abcd x-axis in 1D, in the x-axis and smoothing axis in 2D, or in the smoothing axis integrating out the abcd x-axis in 1D
        self.integrate_shapecorrection_x=False # if the shape correction factor for the abcd x-axis should be inclusive or differential
        self.interpolate_x = True
        self.interpolation_order = 1
        # self.interpolation_order = 2
        self.smooth_shapecorrection = True
        self.smoothing_order_shapecorrection = [1,1] 
        # self.smoothing_order_shapecorrection = [2,1,0]#,1,0]

        # fakerate factor
        self.smooth_fakerate = True
        self.smoothing_order_fakerate=1

        # solve with non negative least squares
        self.solve = solve_nonnegative_leastsquare
        # self.solve = solve_leastsquare

        # set smooth functions
        self.f_scf = None
        self.f_frf = None
        if self.interpolate_x and self.smooth_shapecorrection: 
            self.f_scf = get_smoothing_function_2D(self.interpolation_order, self.smoothing_order_shapecorrection, pol=self.polynomial)
        elif self.interpolate_x:
            self.f_scf = get_smoothing_function_1D(self.interpolation_order, pol=self.polynomial)
        elif self.smooth_shapecorrection:
            self.f_scf = get_smoothing_function_1D(self.smoothing_order_shapecorrection, pol=self.polynomial)            

        if self.smooth_fakerate:
            self.f_frf = get_smoothing_function_1D(self.smoothing_order_fakerate, pol=self.polynomial)
            
        # threshold to warn user in case of large uncertainties
        self.rel_unc_thresh = 1

    # set slices object for selection of sideband regions
    def set_sidebands_selections(self):
        edges_x = self.axis_x.edges
        edges_y = self.axis_y.edges
        self.sel_dx = slice(complex(0, edges_x[1]), complex(0, edges_x[2]), hist.sum)
        self.sel_dy = slice(complex(0, edges_y[1]), complex(0, edges_y[2]), hist.sum)
        if self.name_x in ["mt"]:
            self.sel_x = slice(complex(0, edges_x[2]), None)
            self.sel_d2x = slice(complex(0, edges_x[0]), complex(0, edges_x[1]), hist.sum)
        elif self.name_x in ["dxy", "iso"]:
            self.sel_x = slice(complex(0, edges_x[0]), complex(0, edges_x[1]), hist.sum)
            self.sel_d2x = slice(complex(0, edges_x[2]), None, hist.sum)
        if self.name_y in ["mt"]:
            self.sel_y = slice(complex(0, edges_y[2]), None, hist.sum)
            self.sel_d2y = slice(complex(0, edges_y[0]), complex(0, edges_y[1]), hist.sum)
        elif self.name_y in ["dxy", "iso"]:
            self.sel_y = slice(complex(0, edges_y[0]), complex(0, edges_y[1]), hist.sum)
            self.sel_d2y = slice(complex(0, edges_y[2]), None, hist.sum)

    # TODO diagnostics functions to look at chi2 etc.
    # if return_function:
    #     y_pred = f(x_interpol, x_smoothing, params)
    #     # flatten
    #     y_pred = y_pred.reshape(y.shape) 
    #     w = w.reshape(y.shape)
    #     chi2, ndf = compute_chi2(y, y_pred, w, nparams=params.shape[-1])
    #     return params, cov, f, chi2, ndf

    def get_hist(self, h, variations_scf=False, variations_frf=False, flow=True):
        # TODO: take properly into account fakerate_integration_axes

        if variations_scf and variations_frf:
            raise RuntimeError(f"Can only calculate vairances for fakerate factor or shape correction factor but not both")
        elif variations_scf or variations_frf:
            storage = hist.storage.Double()
        else:
            storage = h.storage_type()

        idx_x = h.axes.name.index(self.name_x)
        if self.smooth_fakerate or self.interpolate_x or self.smooth_shapecorrection:
            y_frf, y_frf_var = self.compute_fakeratefactor(h, variations=variations_frf)
            y_scf, y_scf_var = self.compute_shapecorrection(h, variations=variations_scf)
            c, cy, cvar, cyvar = self.get_yields_applicationregion(h)

            # broadcast abcd-x axis
            new_axis = [np.newaxis if i==idx_x else slice(None) for i in range(c.ndim)]
            y_frf = y_frf[*new_axis] 
            y_frf_var = y_frf_var[*new_axis] if y_frf_var is not None else None
            if self.integrate_shapecorrection_x:
                y_scf = y_scf[*new_axis]
                y_scf_var = y_scf_var[*new_axis] if y_scf_var is not None else None

            d = c * y_scf * y_frf

            if variations_scf and (self.interpolate_x or self.smooth_shapecorrection):
                dvar = c[..., np.newaxis] * y_frf[...,np.newaxis] * y_scf_var[...,:]
                dvar = dvar - d[...,np.newaxis]
            elif variations_frf and self.smooth_fakerate:
                dvar = c[..., np.newaxis] * y_scf[..., np.newaxis] * y_frf_var[...,:]
                dvar = dvar - d[...,np.newaxis]
            elif self.smooth_shapecorrection or self.interpolate_x:
                # only take bin by bin uncertainty from c region
                dvar = (y_scf * y_frf)**2 * cvar
            else:
                # take bin by bin uncertainty from c * c/cy 
                dvar = y_frf**2 *( 4*(c/cy)**2 * cvar + (c/cy**2)**2 * cyvar )

        else:
            # no smoothing
            d, dvar = self.calculate_fullABCD(h)

        if self.integrate_pass_x:
            # set histogram in signal region
            axes = [a for a in h.axes if a.name not in [self.name_x, self.name_y]]
            hSignal = hist.Hist(*axes, storage=storage)
            d = d.sum(axis=idx_x)
            dvar = dvar.sum(axis=idx_x)
        else:
            hSignal = h[{self.name_y: y}] 

        hSignal.values(flow=flow)[...] = d
        if hSignal.storage_type() == hist.storage.Weight:
            hSignal.variances(flow=flow)[...] = dvar

        if (variations_frf and self.smooth_fakerate) or (variations_scf and (self.interpolate_x or self.smooth_shapecorrection)):
            # set systematic histogram with differences between variation and syst
            hsyst = hist.Hist(*axes, hist.axis.Integer(0, dvar.shape[-1], name="_param", overflow=False, underflow=False), storage=storage)

            # limit variations to 0
            vcheck = dvar < (-1*d[...,np.newaxis])
            if np.sum(vcheck) > 0:
                logger.warning(f"Found {np.sum(vcheck)} bins with variations giving negative yields, set these to 0")
                # dvar[vcheck] = 0 

            hsyst.values(flow=flow)[...] = dvar

            # decorrelate in fakerate axes
            axes_names = [n for n in self.fakerate_axes if n != self.smoothing_axis_name]
            hsyst = hh.expand_hist_by_duplicate_axes(hsyst, axes_names, [f"_{n}" for n in axes_names])    

            # add nominal hist and broadcast
            hSignal = hh.addHists(hSignal, hsyst)

        return hSignal

    def compute_shapecorrection(self, h, variations=False, flow=True):
        # rebin in smoothing axis to have stable ratios
        hnew = hh.rebinHist(h, "pt", self.rebin_pt) if self.rebin_pt is not None else h
        hnew = hh.rebinHist(hnew, "pt", self.rebin_x) if self.rebin_x is not None else hnew
        hnew = hnew[{self.name_x: self.sel_x}]
        if self.use_container:
            hnew = hh.transfer_variances(hnew, self.h_shapecorrection)

        c = hnew[{self.name_y: self.sel_dy}].values(flow=flow)
        cy = hnew[{self.name_y: self.sel_d2y}].values(flow=flow)
        if h.storage_type == hist.storage.Weight:
            cvar = hnew[{self.name_y: self.sel_dy}].variances(flow=flow)
            cyvar = hnew[{self.name_y: self.sel_d2y}].variances(flow=flow)

        if self.integrate_shapecorrection_x:
            idx_x = hnew.axes.name.index(self.name_x)
            c = c.sum(axis=idx_x)
            cy = cy.sum(axis=idx_x)
            cvar = cvar.sum(axis=idx_x)
            cyvar = cyvar.sum(axis=idx_x)

        # shape correction factor
        y = c/cy
        mask = (cy <= 0) # masking bins with negative entries
        if mask.sum() > 0:
            logger.warning(f"{(cy < 0)} bins with negative and {(cy == 0)} bins with empty content found for denominator in the shape correction factor.")
        
        if h.storage_type == hist.storage.Weight:
            mask_den = (cyvar**0.5/cy > self.rel_unc_thresh)
            if mask_den.sum() > 0:
                logger.warning(f"{mask_den.sum()} bins with a relative uncertainty larger than {self.rel_unc_thresh*100}% for denominator in the shape correction factor.")
            mask = mask | mask_den

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
            if idx_ax_interpol != len(axes)-1:
                y = np.moveaxis(y, idx_ax_interpol, -1)
                w = np.moveaxis(w, idx_ax_interpol, -1)

            # constract x points in application region
            x_interpol = h[{self.name_x: self.sel_x}].axes[self.name_x].centers
            if h[{self.name_x: self.sel_x}].axes[self.name_x].traits.underflow:
                x_interpol = np.array([x_interpol[0]-np.diff(x_interpol[:2])])
            if h[{self.name_x: self.sel_x}].axes[self.name_x].traits.overflow:
                x_interpol = np.append(x_interpol, x_interpol[-1]+np.diff(x_interpol[-2:]))

            if self.polynomial=="bernstein":
                x_interpol = (x_interpol-self.axis_x_min)/(self.axis_x_max - self.axis_x_min)

            if self.smooth_shapecorrection:
                # interpolate scf in mT and smooth in pT (i.e. 2D)

                idx_ax_smoothing = hnew.axes.name.index(self.smoothing_axis_name)
                if idx_ax_smoothing != len(axes)-2:
                    y = np.moveaxis(y, idx_ax_smoothing, -2)
                    w = np.moveaxis(w, idx_ax_smoothing, -2)

                x_smoothing = hnew.axes[self.smoothing_axis_name].centers
                if self.polynomial=="bernstein":
                    x_smoothing = (x_smoothing-self.smoothing_axis_min)/(self.smoothing_axis_max - self.smoothing_axis_min)

                X, y, XTY = get_parameter_matrices_from2D(x_interpol, x_smoothing, y, w, 
                    self.interpolation_order, self.smoothing_order_shapecorrection, pol=self.polynomial, mask=mask, flatten=True)
                params, cov = self.solve(X, XTY)

                x_smooth = h.axes[self.smoothing_axis_name].centers
                if self.polynomial=="bernstein":
                    x_smooth = (x_smooth-self.smoothing_axis_min)/(self.smoothing_axis_max - self.smoothing_axis_min)
                y_smooth = self.f_scf(x_interpol, x_smooth, params)

                if variations:
                    params_var = get_eigen_variations(cov, params)
                    y_pred_var = self.f_scf(x_interpol, x_smooth, params_var)
                    y_pred_var = np.moveaxis(y_pred_var, params.ndim-1, -1) # put parameter variations last
                else: 
                    y_pred_var = None

                # move interpolation axis to original positon again
                if idx_ax_smoothing != len(axes)-2:
                    y_smooth = np.moveaxis(y_smooth, -2, idx_ax_smoothing)
                    y_pred_var = np.moveaxis(y_pred_var, -3, idx_ax_smoothing) if variations else None
            else:
                # interpolate scf in mT in 1D
                X, XTY = get_parameter_matrices(x_interpol, y, w, self.smoothing_order_fakerate, pol=self.polynomial, mask=mask)
                params, cov = self.solve(X, XTY)

                y_smooth = self.f_scf(x_interpol, params)

                if variations:
                    params_var = get_eigen_variations(cov, params)
                    y_pred_var = self.f_scf(x_interpol, params_var)
                    y_pred_var = np.moveaxis(y_pred_var, params.ndim-1, -1) # put parameter variations last
                else: 
                    y_pred_var = None

            # move interpolation axis to original positon again
            if idx_ax_interpol != len(axes)-1:
                y_smooth = np.moveaxis(y_smooth, -1, idx_ax_interpol)
                y_pred_var = np.moveaxis(y_pred_var, -2, idx_ax_smoothing) if variations else None
    
            # check for negative rates
            if np.sum(y_smooth<0) > 0:
                logger.warning(f"Found {np.sum(y_smooth<0)} bins with negative shape correction factors")
            if y_pred_var is not None and np.sum(y_pred_var<0) > 0:
                logger.warning(f"Found {np.sum(y_pred_var<0)} bins with negative shape correction factor variations")

            return y_smooth, y_pred_var            
        elif self.smooth_shapecorrection:
            # don't interpolate in mT, but smooth in pT in 1D

            # move smoothing axis to last
            axes = [n for n in h.axes.name if n not in [self.name_x, self.name_y] ]
            idx_ax_smoothing = axes.index(self.smoothing_axis_name)
            if idx_ax_smoothing != len(axes)-1:
                y = np.moveaxis(y, idx_ax_smoothing, -1)
                w = np.moveaxis(w, idx_ax_smoothing, -1)
                mask = np.moveaxis(mask, idx_ax_smoothing, -1)

            # smooth scf (e.g. in pT)
            x_smoothing = hnew.axes[self.smoothing_axis_name].centers
            if self.polynomial=="bernstein":
                x_smoothing = (x_smoothing-self.smoothing_axis_min)/(self.smoothing_axis_max - self.smoothing_axis_min)

            X, XTY = get_parameter_matrices(x_smoothing, y, w, self.smoothing_order_shapecorrection, pol=self.polynomial, mask=mask)
            params, cov = self.solve(X, XTY)

            # evaluate in range of original histogram
            x_smooth = h.axes[self.smoothing_axis_name].centers
            if self.polynomial=="bernstein":
                x_smoothing = (x_smoothing-self.smoothing_axis_min)/(self.smoothing_axis_max - self.smoothing_axis_min)

            y_smooth = self.f_scf(x_smooth, params)

            if variations:
                params_var = get_eigen_variations(cov, params)
                y_pred_var = self.f_scf(x_smooth, params_var)
                y_pred_var = np.moveaxis(y_pred_var, params.ndim-1, -1) # put parameter variations last
            else: 
                y_pred_var = None

            # move smoothing axis to original positon again
            if idx_ax_smoothing != len(axes)-1:
                y_smooth = np.moveaxis(y_smooth, -1, idx_ax_smoothing)
                y_pred_var = np.moveaxis(y_pred_var, -2, idx_ax_smoothing) if variations else None

            # check for negative rates
            if np.sum(y_smooth<0) > 0:
                logger.warning(f"Found {np.sum(y_smooth<0)} bins with negative shape correction factors")
            if y_pred_var is not None and np.sum(y_pred_var<0) > 0:
                logger.warning(f"Found {np.sum(y_pred_var<0)} bins with negative shape correction factor variations")

            return y_smooth, y_pred_var        
        else:
            return y, y_var

    def compute_fakeratefactor(self, h, variations=False, flow=True, throw_toys=False):
        # rebin in smoothing axis to have stable ratios
        hnew = hh.rebinHist(h, "pt", self.rebin_pt) if self.rebin_pt is not None else h
        if self.use_container:
            hnew = hh.transfer_variances(hnew, self.h_fakerate)

        # select sideband regions
        a = hnew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].values(flow=flow)
        ax = hnew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].values(flow=flow)
        ay = hnew[{self.name_x: self.sel_dx, self.name_y: self.sel_d2y}].values(flow=flow)
        axy = hnew[{self.name_x: self.sel_d2x, self.name_y: self.sel_d2y}].values(flow=flow)
        b = hnew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].values(flow=flow)
        bx = hnew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].values(flow=flow)

        if h.storage_type == hist.storage.Weight:
            avar = hnew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].variances(flow=flow)
            axvar = hnew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].variances(flow=flow)
            ayvar = hnew[{self.name_x: self.sel_dx, self.name_y: self.sel_d2y}].variances(flow=flow)
            axyvar = hnew[{self.name_x: self.sel_d2x, self.name_y: self.sel_d2y}].variances(flow=flow)
            bvar = hnew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].variances(flow=flow)
            bxvar = hnew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].variances(flow=flow)

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

        mask = (y_den <= 0) # masking bins with negative entries
        if mask.sum() > 0:
            logger.warning(f"{mask.sum()} bins with negative or empty content found for denominator in the fakerate factor.")

        if h.storage_type == hist.storage.Weight:
            # masking bins with large statistical uncertainty (relative uncertainty > 100%)
            y_den_var = y_den**2 * (16*avar/a**2 + axyvar/axy**2 + bxvar/bx**2)
            mask_den = (y_den_var**0.5/y_den > self.rel_unc_thresh)
            if mask_den.sum() > 0:
                logger.warning(f"{mask_den.sum()} bins with a relative uncertainty larger than {self.rel_unc_thresh*100}% for denominator in the fakerate factor.")
            mask = mask | mask_den

            # full variances
            y_var = y**2 * (4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 16*avar/a**2 + bxvar/bx**2)

        if self.smooth_fakerate:
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
                mask = np.moveaxis(mask, idx_ax_smoothing, -1)

            # smooth frf (e.g. in pT)
            x_smoothing = hnew.axes[self.smoothing_axis_name].centers

            if self.polynomial=="bernstein":
                # normalize to [0,1]
                x_smoothing = (x_smoothing-self.smoothing_axis_min)/(self.smoothing_axis_max - self.smoothing_axis_min)

            X, XTY = get_parameter_matrices(x_smoothing, y, w, self.smoothing_order_fakerate, pol=self.polynomial, mask=mask)
            params, cov = self.solve(X, XTY)

            # evaluate in range of original histogram
            x_smooth = h.axes[self.smoothing_axis_name].centers
            if self.polynomial=="bernstein":
                x_smooth = (x_smooth-self.smoothing_axis_min)/(self.smoothing_axis_max - self.smoothing_axis_min)

            y_smooth = self.f_frf(x_smooth, params)

            if variations:
                params_var = get_eigen_variations(cov, params)
                y_pred_var = self.f_frf(x_smooth, params_var)
                y_pred_var = np.moveaxis(y_pred_var, params.ndim-1, -1) # put parameter variations last
            else: 
                y_pred_var = None

            # move smoothing axis to original positon again
            if idx_ax_smoothing != len(axes)-1:
                y_smooth = np.moveaxis(y_smooth, -1, idx_ax_smoothing)
                y_pred_var = np.moveaxis(y_pred_var, -2, idx_ax_smoothing) if variations else None

            # check for negative rates
            if np.sum(y_smooth<0) > 0:
                logger.warning(f"Found {np.sum(y_smooth<0)} bins with negative fake rate factors")
            if y_pred_var is not None and np.sum(y_pred_var<0) > 0:
                logger.warning(f"Found {np.sum(y_pred_var<0)} bins with negative fake rate factor variations")

            return y_smooth, y_pred_var
        else:
            return y, y_var

    def get_yields_applicationregion(self, h, flow=True):
        hC = h[{self.name_x: self.sel_x}]
        c = hC[{self.name_y: self.sel_dy}].values(flow=flow)
        cy = hC[{self.name_y: self.sel_d2y}].values(flow=flow)
        if h.storage_type == hist.storage.Weight:
            cvar = hC[{self.name_y: self.sel_dy}].variances(flow=flow)
            cyvar = hC[{self.name_y: self.sel_d2y}].variances(flow=flow)
            return c, cy, cvar, cyvar
        return c, cy

    def calculate_fullABCD(self, h, flow=True):
        # calculate extended ABCD method without smoothing and in a single bin in x
        a = h[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].values(flow=flow)
        ax = h[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].values(flow=flow)
        ay = h[{self.name_x: self.sel_dx, self.name_y: self.sel_d2y}].values(flow=flow)
        axy = h[{self.name_x: self.sel_d2x, self.name_y: self.sel_d2y}].values(flow=flow)
        b = h[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].values(flow=flow)
        bx = h[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].values(flow=flow)
        c = h[{self.name_x: self.sel_x, self.name_y: self.sel_dy}].values(flow=flow)
        cy = h[{self.name_x: self.sel_x, self.name_y: self.sel_d2y}].values(flow=flow)
        d = ((ax*ay*b)[...,np.newaxis]*c)**2 / ((a**4 * axy*bx)[...,np.newaxis]*cy)
        if h.storage_type == hist.storage.Weight:
            avar = h[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}].variances(flow=flow)
            axvar = h[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}].variances(flow=flow)
            ayvar = h[{self.name_x: self.sel_dx, self.name_y: self.sel_d2y}].variances(flow=flow)
            axyvar = h[{self.name_x: self.sel_d2x, self.name_y: self.sel_d2y}].variances(flow=flow)
            bvar = h[{self.name_x: self.sel_dx, self.name_y: self.sel_y}].variances(flow=flow)
            bxvar = h[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}].variances(flow=flow)
            cvar = h[{self.name_x: self.sel_x, self.name_y: self.sel_dy}].variances(flow=flow)
            cyvar = h[{self.name_x: self.sel_x, self.name_y: self.sel_d2y}].variances(flow=flow)
            dvar = d**2 * ((4*axvar/ax**2 + 4*ayvar/ay**2 + axyvar/axy**2 + 4*bvar/b**2 + 16*avar/a**2 + bxvar/bx**2)[...,np.newaxis] + 4*cvar/c**2 + cyvar/cy**2 )
            return d, dvar
        return d
