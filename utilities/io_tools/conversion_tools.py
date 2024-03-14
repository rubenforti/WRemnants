
import itertools
import hist
import numpy as np

from narf import ioutils

from utilities import logging
from utilities.io_tools import combinetf_input

logger = logging.child_logger(__name__)

def fitresult_pois_to_hist(infile, poi_types = ["mu", "pmaskedexp", "pmaskedexpnorm", "sumpois", "sumpoisnorm", ], grouped=True, uncertainties=None):
    # convert POIs in fitresult into histograms
    # uncertainties, use None to get all, use [] to get none
    # grouped=True for grouped uncertainties 
    fitresult = combinetf_input.get_fitresult(infile)
    meta = ioutils.pickle_load_h5py(fitresult["meta"])
    meta_info = meta["meta_info"]

    result = {}
    for poi_type in poi_types:
        logger.debug(f"Now at POI type {poi_type}")

        scale = 1 
        if poi_type in ["nois"]:
            scale = 1./(imeta["args"]["scaleNormXsecHistYields"]*imeta["args"]["priorNormXsec"])

        df = combinetf_input.read_impacts_pois(fitresult, poi_type=poi_type, group=grouped, uncertainties=uncertainties)

        result[poi_type] = {}
        for channel, gen_axes in meta["channel_gen_axes"].items():
            logger.debug(f"Now at channel {channel}")

            channel_scale = scale
            if poi_type in ["pmaskedexp", "sumpois"]:
                channel_scale = 1000*meta["channel_lumi"][channel]

            result[poi_type][channel] = {}
            for proc, gen_axes_proc in gen_axes.items():
                logger.debug(f"Now at proc {proc}")

                if poi_type.startswith("sum"):
                    # make all possible lower dimensional gen axes combinations; wmass only combinations including qGen
                    gen_axes_permutations = [list(k) for n in range(1, len(gen_axes_proc)) for k in itertools.combinations(gen_axes_proc, n)]
                else:
                    gen_axes_permutations = [gen_axes_proc[:],]

                result[poi_type][channel][proc] = {}
                for axes in gen_axes_permutations:
                    shape = [a.extent for a in axes]
                    axes_names = [a.name for a in axes]

                    data = combinetf_input.select_pois(df, axes_names, base_processes=proc, flow=True)

                    values = np.reshape(data["value"].values/channel_scale, shape)
                    variances = np.reshape( (data["err_total"].values/channel_scale)**2, shape)

                    h_ = hist.Hist(*axes, storage=hist.storage.Weight())
                    h_.view(flow=True)[...] = np.stack([values, variances], axis=-1)

                    hist_name = "hist_" + "_".join(axes_names)
                    logger.debug(f"Save histogram {hist_name}")
                    result[poi_type][channel][proc][hist_name] = h_

                    if "err_stat" in data.keys():
                        # save stat only hist
                        variances = np.reshape( (data["err_stat"].values/channel_scale)**2, shape)
                        h_stat = hist.Hist(*axes, storage=hist.storage.Weight())
                        h_stat.view(flow=True)[...] = np.stack([values, variances], axis=-1)
                        result[poi_type][channel][proc][f"{hist_name}_stat"] = h_stat

                    # save other systematic uncertainties as separately varied histograms
                    labels = [u.replace("err_","") for u in filter(lambda x: x.startswith("err_") and x not in ["err_total", "err_stat"], data.keys())]
                    if labels:
                        systs = np.stack([values,*[values + np.reshape(data[f"err_{u}"].values/channel_scale, shape) for u in labels]], axis=-1)
                        h_syst = hist.Hist(*axes, hist.axis.StrCategory(["nominal", *labels], name="syst"), storage=hist.storage.Double())
                        h_syst.values(flow=False)[...] = systs
                        result[poi_type][channel][proc][f"{hist_name}_syst"] = h_syst

    return result, meta