import argparse
from utilities.io_tools import combinetf_input, input_tools, output_tools
from utilities import common
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from wremnants import plot_tools

parser = common.plot_parser()
parser.add_argument("-i", "--fitresult", type=str, required=True, help="fitresults file from combinetf")
args = parser.parse_args()

dfw = pd.DataFrame.from_dict({
    "Name" : ["LEP Combination", "D0", "CDF", "LHCb", "ATLAS", "PDG Average"],
    "value" : [80376, 80375, 80433.5, 80354, 80366.5, 80369.2],
    "err_total" : [33, 23, 9.4, 32, 15.9, 13.3],
    "err_stat" : [25, 11, 6.4, 23, 9.8, 13.3],
    "Reference" : [
        "Phys. Rep. 532 (2013) 119", 
        # "Phys. Rev. Lett. 108 (2012) 151804",
        "PRL 108 (2012) 151804",
        "Science 376 (2022) 6589", 
        "JHEP 01 (2022) 036", 
        "arxiv:2403.15085 Subm. to EPJC", 
        # "Eur.Phys.J.C 84 (2024) 5, 451",
        "EPJC 84 (2024) 5, 451",
    ],
    "color" : ["black"]*5+["navy"],
})

cms_res = combinetf_input.read_groupunc_df(args.fitresult, ["stat",], name="CMS")
cms_res["color"] = "#E42536" 
cms_res["Reference"] = "This Work" 
dfw_cms = pd.concat((dfw, cms_res), ignore_index=True)

eoscp = output_tools.is_eosuser_path(args.outpath)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

nentries = len(dfw_cms)

fig = plot_tools.make_summary_plot(80353, 6, "EW fit",
    dfw_cms.loc[:,("Name", "value", "err_total")],
    colors=list(dfw_cms["color"]),
    xlim=[80255, 80465],
    ylim=[0, nentries+1.3],
    xlabel=r"$\mathit{m}_{W}$ (MeV)", 
    capsize=6,
    width_scale=1.25,
    cms_label=args.cmsDecor,
    cms_loc=0,
    padding=4,
    point_size=0.24,
    #top_offset=offset,
    #bottom_offset=offset*2,
    label_points=False,
    legend_loc='lower right',
    legtext_size="small",
    logoPos=args.logoPos,
)


top = nentries#+0.5
# step = (top+0.25)/nentries
step = top/nentries

ax = plt.gca()
xpos = 80135

text_size = 15 #
text_size_large = plot_tools.get_textsize(ax, "small")

ax.annotate("$\mathit{m}_{{W}}\pm$ unc. in MeV", (80265, top+0.5), fontsize=text_size, ha="left", color="black", annotation_clip=False)
for i,row in dfw_cms.iterrows():
    isCMS = row.loc["Name"] == "CMS" 
    pos = top-step*i
    ax.annotate(row["Name"], (xpos, pos), fontsize=text_size_large, ha="left", annotation_clip=False, color=row.loc["color"])#, weight=600)
    if row.loc["Name"] in ["CMS", "CDF", "ATLAS", "PDG Average"]:
        label = f"{row.loc['value']:.1f} $\pm$ {round(row.loc['err_total'], 1):.1f}"
    else:
        label = f"{row.loc['value']:.0f} $\pm$ {round(row.loc['err_total'], 0):.0f}"
    
    # label = $\mathit{m}_{{W}}$ = "+label+" MeV"

    ax.annotate(label, (80265, pos), fontsize=text_size, ha="left", va="center", color="black", annotation_clip=False)
    ax.annotate(row["Reference"], (xpos, pos-0.42), fontsize=text_size, ha="left", color="dimgrey", annotation_clip=False, style='italic' if isCMS else None)

ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.xaxis.grid(False, which='both')
ax.yaxis.grid(False, which='both')

name = "resultsSummary"
if args.postfix:
    name += postfix
if args.cmsDecor == "Preliminary":
    name += "_preliminary"

plot_tools.save_pdf_and_png(outdir, name, fig)
plot_tools.write_index_and_log(outdir, name)
if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
