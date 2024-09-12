import argparse
from utilities.io_tools import combinetf_input, input_tools, output_tools
from utilities import common
import matplotlib.pyplot as plt
import pandas as pd
from wremnants import plot_tools

parser = common.plot_parser()
parser.add_argument("-i", "--fitresult", type=str, required=True, help="fitresults file from combinetf")
args = parser.parse_args()

dfw = pd.DataFrame.from_dict({
    "Name" : ["LEP Combination", "D0 (Run 2)", "CDF (Run 2)", "LHCb", "ATLAS (2024)", "PDG Average"],
    "value" : [80376, 80375, 80434, 80354, 80367, 80369.2],
    "err_total" : [33, 23, 9.4, 32, 15.9, 13.3],
    "err_stat" : [25, 11, 6.4, 23, 9.8, 13.3],
    "Reference" : ["Phys. Rep. 532 (2013) 119", "Phys. Rev. Lett. 108 (2012) 151804",
                  "Science 376 (2022) 6589", "JHEP 01 (2022) 036", "arxiv:2403.15085 Submitted to EPJC", 
                   "Eur.Phys.J.C 84 (2024) 5, 451"],
    "color" : ["black"]*5+["gray"],
})

cms_res = combinetf_input.read_groupunc_df(args.fitresult, ["stat",], name="CMS")
cms_res["color"] = "#E42536" 
cms_res["Reference"] = "This Work" 
dfw_cms = pd.concat((dfw, cms_res), ignore_index=True)

name = "resultsSummary"
if args.postfix:
    name += postfix

eoscp = output_tools.is_eosuser_path(args.outpath)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

fig = plot_tools.make_summary_plot(80353, 6, None,
    dfw_cms,
    colors=list(dfw_cms["color"]),
    xlim=[80260, 80460],
    xlabel="$m_{W}$ (MeV)", 
    out=outdir, 
    outfolder=args.outfolder,
    name=name,
    legend_loc=None,
    capsize=6,
    width_scale=1.25,
    cms_label=args.cmsDecor,
)

nentries = len(dfw_cms)

ax = plt.gca()
for i,row in dfw_cms.iterrows():
    isCMS = row.loc["Name"] == "CMS" 
    offset = nentries+0.4
    pos = offset-i*1.1
    ax.annotate(row.loc["Name"], (80170, pos), fontsize=18, ha="left", annotation_clip=False, color=row.loc["color"], weight=600)
    ax.annotate(row.loc["Reference"], (80170, pos-0.3), fontsize=10, ha="left", color="dimgrey", annotation_clip=False, style='italic' if isCMS else None)
    label = f"$m_{{W}}$ = {row.loc['value']:.0f} $\pm$ {round(row.loc['err_total'], 0):.0f}"
    if row.loc["Name"] in ["CMS", "CDF"]:
        label = f"$m_{{W}}$ = {row.loc['value']:.1f} $\pm$ {round(row.loc['err_total'], 1):.1f}"
    ax.annotate(label, (80170, pos-0.6), fontsize=10, ha="left", color="dimgrey", annotation_clip=False)

plot_tools.save_pdf_and_png(outdir, name, fig)
plot_tools.write_index_and_log(outdir, name)
if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
