import argparse
from utilities import common, boostHistHelpers as hh
from utilities.io_tools import input_tools, output_tools, combinetf2_input
from wremnants import syst_tools,plot_tools
import hist
import numpy as np
import pickle

parser = common.plot_parser()
parser.add_argument("--unfolded", type=str, required=False)
parser.add_argument("--gen", type=str, required=True)
parser.add_argument("-w", action='store_true')
parser.add_argument("--ptll-fit", type=str, required=True)
parser.add_argument("--etapt-fit", type=str, required=True)
parser.add_argument("--obs", type=str, default="ptVgen")

args = parser.parse_args()

def hist_to_up_down_unc(h):
    hunc = h.copy()
    hunc.values(flow=True)[...] = np.sqrt(h.variances(flow=True))
    hup = hh.addHists(h, hunc)
    hdown = hh.addHists(h, hunc, scale2=-1)
    return hup, hdown

def quadrature_sum_hist(hists, is_down):
    sumh = hist.Hist(*hists[0].axes, hist.axis.Integer(0, len(hists), name="vars"),
                data=np.stack(hists, axis=-1))
    return hh.rssHists(sumh, syst_axis="vars")[is_down]

hnom = "nominal_gen"

unfolded_data = pickle.load(open(args.unfolded, "rb")) if args.unfolded else None

procs = ["WplusmunuPostVFP", "WminusmunuPostVFP"] if args.w else ["ZmumuPostVFP"] 

gen = input_tools.read_all_and_scale(args.gen, procs, [hnom])[0]

scetlib_dyturbo = input_tools.read_all_and_scale(args.gen, procs, [f"{hnom}_scetlib_dyturboCorr"])[0].project(args.obs, "vars")
ct18z = input_tools.read_all_and_scale(args.gen, procs, [f"{hnom}_pdfCT18Z"])[0].project(args.obs, "pdfVar")
ct18z_as = input_tools.read_all_and_scale(args.gen, procs, [f"{hnom}_pdfCT18ZalphaS002"])[0].project(args.obs, "alphasVar")

# Leaving out mb and mc range since they're small...

transforms = syst_tools.syst_transform_map(hnom, f"{hnom}_scetlib_dyturboCorr")

theory_up = quadrature_sum_hist([
        transforms["resumTNPXp0Up"]["action"](scetlib_dyturbo), 
        transforms["resumNPUp"]["action"](scetlib_dyturbo), 
        scetlib_dyturbo[{"vars" : "renorm_scale_pt20_envelope_Up"}],
        scetlib_dyturbo[{"vars" : "transition_points0.2_0.35_1.0"}],
        transforms["pdfCT18ZUp"]["action"](ct18z),
        ct18z_as[{"alphasVar" : "as0120"}],
    ], is_down=False
)

theory_down = quadrature_sum_hist([
        transforms["resumTNPXp0Down"]["action"](scetlib_dyturbo), 
        transforms["resumNPDown"]["action"](scetlib_dyturbo), 
        scetlib_dyturbo[{"vars" : "renorm_scale_pt20_envelope_Down"}],
        scetlib_dyturbo[{"vars" : "transition_points0.2_0.75_1.0"}],
        transforms["pdfCT18ZDown"]["action"](ct18z),
        ct18z_as[{"alphasVar" : "as0116"}],
    ], is_down=True
)

ptll_fit = combinetf2_input.get_fitresult(args.ptll_fit)
etapt_fit = combinetf2_input.get_fitresult(args.etapt_fit)

ptllh = ptll_fit["hist_postfit_inclusive"]["ch0"].get()/1000.
etapth = etapt_fit["hist_postfit_inclusive"]["ch0"].get()/1000.

hists = [x.project(args.obs) for x in [
    gen,
    ptllh,
    etapth,
    theory_up,
    theory_down,
    *hist_to_up_down_unc(ptllh),
    *hist_to_up_down_unc(etapth),
]]


labels=[
        r"prefit",
        r"$m_{W}$ $(p_{T}^{\mu}, \eta^{\mu})+p_{T}^{\mu\mu}$ postfit" if args.w else r"$p_{T}^{\mu\mu}$ postfit",
        ("$m_{W}$ " if args.w else "$m_{Z}$ ")+ r"$(p_{T}^{\mu}, \eta^{\mu})$ postfit",
        "", "", 
        "", "",
        "", "",
        ]
colors=[
        "black",
        "#5790FC",
        "#E42536" if args.w else "#964A8B",
        "gray",
        "gray",
        "#5790FC",
        "#5790FC",
        "#E42536" if args.w else "#964A8B",
        "#E42536" if args.w else "#964A8B",
]

if unfolded_data:
    unfoldedh = unfolded_data["results"]['pmaskedexp']['chan_13TeV']["Z"]["hist_ptVGen"]
    # Hack for now to work around the fact that I used 1 GeV bins here previously
    if args.obs:
        unfoldedh = hh.rebinHist(unfoldedh, "ptVGen", gen.axes["ptVgen"].edges[:21])

    # Thanks Obama (David)
    for ax in unfoldedh.axes:
        ax._ax.metadata["name"] = ax.name.replace("Gen", "gen")
    hists.insert(3, unfoldedh)
    labels.insert(3, "Unfolded data")
    colors.insert(3, "black")

if args.xlim:
    hists = [x[complex(0, args.xlim[0]):complex(0, args.xlim[1])] for x in hists]

hists_nominals = hists[:3+(unfolded_data is not None)]

if args.w:
    xlabel=r'$\mathit{p}_{T}^{W}$ (GeV)'
    ylabel = r'$W\to\mu\nu'
else:
    xlabel=r'$\mathit{p}_{T}^{Z}$ (GeV)'
    ylabel = r'$Z\to\mu\mu'
ylabel += r'\ Cross\ section\,/\,GeV$'

fig = plot_tools.makePlotWithRatioToRef(
    hists=hists_nominals,
    hists_ratio=hists,
    labels=labels,
    colors=colors,
    linestyles=["solid",]*(4 if unfolded_data else 3)+["dotted"]*2+["dashdot"]*2+["dashed"]*2,
    xlabel=xlabel, ylabel=ylabel,
    rlabel="Ratio to prefit",
    rrange=args.rrange,
    nlegcols=args.legCols,
    lowerLegCols=args.lowerLegCols,
    lowerLegPos=args.lowerLegPos,
    yscale=args.yscale,
    ylim=args.ylim,
    xlim=None, binwnorm=1.0, baseline=True,
    yerr=False,
    fill_between=6,
    cms_label=args.cmsDecor,
    legtext_size=args.legSize,
    dataIdx=3 if unfolded_data else None,
    width_scale=1.25,
)
eoscp = output_tools.is_eosuser_path(args.outpath)

outdir = output_tools.make_plot_dir(args.outpath, "Z", eoscp=True)
name = f"ptVgen_postfit_{'W' if args.w else 'Wlike'}_RecoPtll_PrefitRatio"
if args.cmsDecor == "Preliminary":
    name += "_preliminary"

plot_tools.save_pdf_and_png(outdir, name)
plot_tools.write_index_and_log(outdir, name)
if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, "W" if args.w else "Z", deleteFullTmp=True)
