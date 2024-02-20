import numpy as np
from utilities import boostHistHelpers as hh
from utilities import common, logging

from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import unumpy as unp

logger = logging.child_logger(__name__)

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
