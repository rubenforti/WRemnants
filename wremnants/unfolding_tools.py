from utilities import differential
from wremnants import syst_tools, theory_tools, logging
from copy import deepcopy
import hist
import numpy as np
import pandas as pd
import h5py
import uproot

logger = logging.child_logger(__name__)

def add_out_of_acceptance(datasets, group, newGroupName=None):
    # Copy datasets from specified group to make out of acceptance contribution
    datasets_ooa = []
    for dataset in datasets:
        if dataset.group == group:
            ds = deepcopy(dataset)

            if newGroupName is None:
                ds.group = ds.group+"OOA"
            else:
                ds.group = newGroupName
            ds.out_of_acceptance = True

            datasets_ooa.append(ds)

    return datasets + datasets_ooa

def define_gen_level(df, gen_level, dataset_name, mode="wmass"):
    # gen level definitions
    gen_levels = ["preFSR", "postFSR"]
    if gen_level not in gen_levels:
        raise ValueError(f"Unknown gen level '{gen_level}'! Supported gen level definitions are '{gen_levels}'.")

    logger.info(f"Using {gen_level} leptons")
    mz = "mz" in mode

    if gen_level == "preFSR":
        df = theory_tools.define_prefsr_vars(df, mode=mode)

        # needed for fiducial phase space definition
        df = df.Alias("massVGen", "massVgen")
        df = df.Alias("ptVGen", "ptVgen")
        df = df.Alias("absYVGen", "absYVgen")

        if "mw" in mode or "singlelep" in mode:
            df = df.Alias("mTVGen", "mTVgen")   

        if "mw" in mode:
            df = df.Define("ptGen", "chargeVgen < 0 ? genl.pt() : genlanti.pt()")   
            df = df.Define("absEtaGen", "chargeVgen < 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())")

        if mz:
            df = df.Define("ptGen", "event % 2 == 0 ? genl.pt() : genlanti.pt()")
            df = df.Define("absEtaGen", "event % 2 == 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())")
            df = df.Define("ptOtherGen", "event % 2 == 0 ? genlanti.pt() : genl.pt()")
            df = df.Define("absEtaOtherGen", "event % 2 == 0 ? std::fabs(genlanti.eta()) : std::fabs(genl.eta())")

    elif gen_level == "postFSR":
        df = theory_tools.define_postfsr_vars(df, mode=mode)

        df = df.Alias("ptGen", f"postfsrLep_pt")
        df = df.Alias("absEtaGen", f"postfsrLep_absEta")           

        if "mw" in mode or "singlelep" in mode:
            df = df.Alias("mTWGen", "postfsrMT")   
   
        if mz:
            df = df.Alias("ptOtherGen", "postfsrOtherLep_pt")
            df = df.Alias("absEtaOtherGen", f"postfsrOtherLep_absEta")                

            df = df.Alias("massVGen", "postfsrMV")
            df = df.Define("absYVGen", "postfsrabsYV")  

        df = df.Alias("ptVGen", "postfsrPTV")      

    if "wlike" in mode:
        df = df.Define("qGen", "event % 2 == 0 ? -1 : 1")

    return df

def get_fiducial_args(mode, pt_min=28, pt_max=60, abseta_max=2.4):
    fidargs = {}
    if "inclusive" in mode or mode == "mz_masswindow":
        fidargs = {"abseta_max" : 100.}
        if mode == "mz_inclusive":
            fidargs.update({"mass_min" : 60, "mass_max" : 120})
        return fidargs

    fidargs.update({"mtw_min" : 40 if "mw" in mode or "singlelep" in mode else 0,
                    "pt_min" : pt_min, "pt_max" : pt_max, "abseta_max" : abseta_max})

    return fidargs

def select_fiducial_space(df, select=True, accept=True, mode="mw", pt_min=0, pt_max=1300, abseta_max=2.4, mass_min=60, mass_max=120, mtw_min=0, selections=[]):
    # Define a fiducial phase space and if select=True, either select events inside/outside
    # accept = True: select events in fiducial phase space 
    # accept = False: reject events in fiducial pahse space
    
    if "mw" in mode:
        selection = f"(absEtaGen < {abseta_max})"        
    elif "singlelep" in mode:
        selection = f"""
            (absEtaGen < {abseta_max}) && (absEtaOtherGen < {abseta_max}) 
            && (ptOtherGen > {pt_min}) && (ptOtherGen < {pt_max})
            && (massVGen > {mass_min}) && (massVGen < {mass_max})
            """
    elif "mz" in mode:
        selection = f"""
            (absEtaGen < {abseta_max}) && (absEtaOtherGen < {abseta_max}) 
            && (ptGen > {pt_min}) && (ptOtherGen > {pt_min})
            && (ptGen < {pt_max}) && (ptOtherGen < {pt_max})
            && (massVGen > {mass_min}) && (massVGen < {mass_max})
        """
    else:
        raise NotImplementedError(f"No fiducial phase space definiton found for mode '{mode}'!") 

    if mtw_min > 0:
        selection += f" && (mTVGen > {mtw_min})"

    for sel in selections:
        logger.debug(f"Add selection {sel} for fiducial phase space")
        selection += f" && ({sel})"

    logger.info(f"Applying fiducial selection {selection}")

    df = df.Define("acceptance", selection)

    if select and accept:
        df = df.Filter("acceptance")
    elif select :
        df = df.Filter("acceptance == 0")

    return df

def add_xnorm_histograms(results, df, args, dataset_name, corr_helpers, qcdScaleByHelicity_helper, unfolding_axes, unfolding_cols):
    # add histograms before any selection
    df_xnorm = df
    df_xnorm = df_xnorm.DefinePerSample("exp_weight", "1.0")

    df_xnorm = theory_tools.define_theory_weights_and_corrs(df_xnorm, dataset_name, corr_helpers, args)

    df_xnorm = df_xnorm.Define("xnorm", "0.5")

    axis_xnorm = hist.axis.Regular(1, 0., 1., name = "count", underflow=False, overflow=False)

    xnorm_axes = [axis_xnorm, *unfolding_axes]
    xnorm_cols = ["xnorm", *unfolding_cols]
    
    results.append(df_xnorm.HistoBoost("xnorm", xnorm_axes, [*xnorm_cols, "nominal_weight"]))

    syst_tools.add_theory_hists(results, df_xnorm, args, dataset_name, corr_helpers, qcdScaleByHelicity_helper, xnorm_axes, xnorm_cols, base_name="xnorm")

