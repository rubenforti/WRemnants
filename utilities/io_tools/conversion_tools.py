
import itertools
import hist
import numpy as np

from narf import ioutils

from utilities import logging
from utilities.io_tools import combinetf_input

logger = logging.child_logger(__name__)

def transform_poi(poi_type, meta):
    # scale poi type, in case of noi, convert into poi like form
    if poi_type in ["nois"]:
        prior_norm = meta["args"]["priorNormXsec"]
        scale_norm = meta["args"]["scaleNormXsecHistYields"] if meta["args"]["scaleNormXsecHistYields"] is not None else 1
        val = lambda theta, scale, k=prior_norm, x=scale_norm: (1+x)**(k * theta) * scale 
        err = lambda theta, err, scale, k=prior_norm, x=scale_norm: (k*np.log(1+x) * (1+x)**(k * theta)) * scale * err
        
        # val = lambda theta, scale, k=prior_norm, x=scale_norm: (1 + x * k * theta) * scale 
        # err = lambda theta, scale, k=prior_norm, x=scale_norm: (x * k * theta) * scale 
        return val, err
    else:
        val = lambda poi, scale: poi * scale
        err = lambda poi, err, scale: err * scale
        return val, err

def fitresult_pois_to_hist(infile, result=None, poi_types = None, translate_poi_types=True,
    merge_channels=True, grouped=True, uncertainties=None, expected=False, initial=None, flow=True,
):
    # convert POIs in fitresult into histograms
    # uncertainties, use None to get all, use [] to get none
    # grouped=True for grouped uncertainties 
    # Different channels can have different year, flavor final state, particle final state, sqrt(s), 
    #   if merge_channels=True the lumi is added up for final states with different flavors or eras with same sqrt(s)
    # initial: results from an initial fit to set noi values
    
    # translate the name of the keys to be written out
    target_keys={
        "pmaskedexp": "xsec",
        "sumpois": "xsec",
        "pmaskedexpnorm": "xsec_normalized",
        "sumpoisnorm": "xsec_normalized",
    }
    
    channel_energy={
        "2017G": "5TeV",
        "2017H": "13TeV",
        "2016preVFP": "13TeV", 
        "2016postVFP":"13TeV", 
        "2017": "13TeV", 
        "2018": "13TeV",
    }
    channel_flavor ={
        "e": "l", 
        "mu": "l",
        "ee": "ll", 
        "mumu": "ll",
    }

    fitresult = combinetf_input.get_fitresult(infile.replace(".root",".hdf5"))
    meta = ioutils.pickle_load_h5py(fitresult["meta"])
    meta_info = meta["meta_info"]

    if initial:
        fitresult_initial = combinetf_input.get_fitresult(initial.replace(".root",".hdf5"))
        meta_initial = ioutils.pickle_load_h5py(fitresult_initial["meta"])
        meta_info_initial = meta_initial["meta_info"]

    if poi_types is None:
        if meta_info["args"]["poiAsNoi"]:
            poi_types = ["nois"]
        else:
            poi_types = ["mu", "pmaskedexp", "pmaskedexpnorm", "sumpois", "sumpoisnorm"]
    
    if merge_channels:
        channel_info = {}
        for chan, info in meta["channel_info"].items():
            if chan.endswith("masked"):
                continue
            channel = f"chan_{channel_energy[info['era']]}"
            if info['flavor'] in channel_flavor:
                channel += f"_{channel_flavor[info['flavor']]}"

            logger.debug(f"Merge channel {chan} into {channel}")

            lumi = info["lumi"]
            gen_axes = info["gen_axes"]    

            if channel not in channel_info:
                channel_info[channel] = {
                    "gen_axes": gen_axes,
                    "lumi": lumi,
                }
            else:
                if gen_axes != channel_info[channel]["gen_axes"]:
                    raise RuntimeError(f"The gen axes are different among channels {channel_info}, so they can't be merged")
                channel_info[channel]["lumi"] += lumi
    else:
        channel_info = meta["channel_info"]

    if result is None: 
        result = {}
    for poi_type in poi_types:
        logger.debug(f"Now at POI type {poi_type}")

        df = combinetf_input.read_impacts_pois(fitresult, poi_type=poi_type, group=grouped, uncertainties=uncertainties)
        if df is None or len(df)==0:
            logger.warning(f"POI type {poi_type} not found in histogram, continue with next one")
            continue

        action_val, action_err = transform_poi(poi_type, meta_info)

        poi_type_initial="mu"
        if initial:
            action_val_initial, action_err_initial = transform_poi(poi_type_initial, meta_info_initial)
            df_initial = combinetf_input.read_impacts_pois(fitresult_initial, poi_type=poi_type_initial, group=grouped, uncertainties=[])
            if df_initial is None or len(df_initial)==0:
                logger.warning(f"Initial fitresult specified but POI type {poi_type_initial} not found in histogram, skip using initial result")

        poi_key = target_keys.get(poi_type, poi_type) if translate_poi_types else poi_type
        if poi_key not in result:
            result[poi_key] = {}
        for channel, info in channel_info.items():
            logger.debug(f"Now at channel {channel}")

            if poi_type in ["pmaskedexp", "sumpois"]:
                channel_scale = info["lumi"]/1000
                channel_scale_initial = info["lumi"]/1000
            else:
                channel_scale = 1
                channel_scale_initial = 1
            if channel not in result[poi_key]:
                result[poi_key][channel] = {}
            for proc, gen_axes_proc in info["gen_axes"].items():
                logger.debug(f"Now at proc {proc}")

                if poi_type.startswith("sum"):
                    if len(gen_axes_proc)==1:
                        logger.info(f"Skip POI type {poi_type} since there is only one gen axis")
                        continue
                    # make all possible lower dimensional gen axes combinations; wmass only combinations including qGen
                    gen_axes_permutations = [list(k) for n in range(1, len(gen_axes_proc)) for k in itertools.combinations(gen_axes_proc, n)]
                else:
                    gen_axes_permutations = [gen_axes_proc[:],]

                if proc not in result[poi_key][channel]:
                    result[poi_key][channel][proc] = {}
                for axes in gen_axes_permutations:
                    shape = [a.extent if flow else a.size for a in axes]
                    axes_names = [a.name for a in axes]

                    data = combinetf_input.select_pois(df, axes_names, base_processes=proc, flow=flow)
                    for u in filter(lambda x: x.startswith("err_"), data.keys()):
                        data.loc[:,u] = action_err(data["value"], data[u], channel_scale)
                    data.loc[:,"value"] = action_val(data["value"], channel_scale)
                    logger.debug(f"The values for the hist in poi {poi_key} are {data['value'].values}")

                    if initial:
                        data_initial = combinetf_input.select_pois(df_initial, axes_names, base_processes=proc, flow=flow)
                        values_initial = action_val_initial(data_initial["value"].values, channel_scale_initial)
                        data.loc[:,"value"] = data["value"] * values_initial 
                        for u in filter(lambda x: x.startswith("err_"), data.keys()):
                            data.loc[:,u] = data[u] * values_initial

                    values = np.reshape(data["value"].values, shape)
                    variances = np.reshape(data["err_total"].values**2, shape)

                    # save nominal histogram with total uncertainty
                    h_ = hist.Hist(*axes, storage=hist.storage.Weight())
                    h_.view(flow=flow)[...] = np.stack([values, variances], axis=-1)

                    hist_name = "hist_" + "_".join(axes_names)
                    if expected:
                        hist_name+= "_expected"
                    logger.info(f"Save histogram {hist_name}")
                    if hist_name in result[poi_key][channel][proc]:
                        logger.warning(f"Histogram {hist_name} already in result, it will be overridden")
                    result[poi_key][channel][proc][hist_name] = h_

                    # save nominal histogram with stat uncertainty
                    if "err_stat" in data.keys():
                        variances = np.reshape( (data["err_stat"].values)**2, shape)
                        h_stat = hist.Hist(*axes, storage=hist.storage.Weight())
                        h_stat.view(flow=flow)[...] = np.stack([values, variances], axis=-1)
                        hist_name_stat = f"{hist_name}_stat"
                        if hist_name_stat in result[poi_key][channel][proc]:
                            logger.warning(f"Histogram {hist_name_stat} already in result, it will be overridden")
                        result[poi_key][channel][proc][hist_name_stat] = h_stat

                    # save other systematic uncertainties as separately varied histograms
                    labels = [u.replace("err_","") for u in filter(lambda x: x.startswith("err_") and x not in ["err_total", "err_stat"], data.keys())]
                    if labels:
                        # string category always has an overflow bin, we set it to 0
                        systs = np.stack([values, *[values + np.reshape(data[f"err_{u}"].values, shape) for u in labels]], axis=-1)
                        if flow:
                            systs = np.stack([systs, np.zeros_like(values)], axis=-1) # overflow axis can't be disabled in StrCategory
                        h_syst = hist.Hist(*axes, hist.axis.StrCategory(["nominal", *labels], name="syst"), storage=hist.storage.Double())
                        h_syst.values(flow=flow)[...] = systs
                        hist_name_syst = f"{hist_name}_syst"
                        if hist_name_syst in result[poi_key][channel][proc]:
                            logger.warning(f"Histogram {hist_name_syst} already in result, it will be overwritten")
                        result[poi_key][channel][proc][hist_name_syst] = h_syst

    return result, meta
