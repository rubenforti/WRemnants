import argparse
from utilities import common, rdf_tools, logging, differential
from utilities.io_tools import output_tools
from utilities.common import background_MCprocs as bkgMCprocs
from wremnants.datasets.datagroups import Datagroups

parser,initargs = common.common_parser(True)

import ROOT
import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections, muon_calibration, muon_selections, muon_validation, unfolding_tools, theoryAgnostic_tools, helicity_utils
from wremnants.histmaker_tools import scale_to_data, aggregate_groups
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.helicity_utils_polvar import makehelicityWeightHelper_polvar 
import hist
import lz4.frame
import math
import time
from utilities import boostHistHelpers as hh
import pathlib
import os
import numpy as np

data_dir = common.data_dir
parser.add_argument("--lumiUncertainty", type=float, help="Uncertainty for luminosity in excess to 1 (e.g. 1.012 means 1.2\%)", default=1.012)
parser.add_argument("--noGenMatchMC", action='store_true', help="Don't use gen match filter for prompt muons with MC samples (note: QCD MC never has it anyway)")
parser.add_argument("--halfStat", action='store_true', help="Test half data and MC stat, selecting odd events, just for tests")
parser.add_argument("--makeMCefficiency", action="store_true", help="Save yields vs eta-pt-ut-passMT-passIso-passTrigger to derive 3D efficiencies for MC isolation and trigger (can run also with --onlyMainHistograms)")
parser.add_argument("--onlyTheorySyst", action="store_true", help="Keep only theory systematic variations, mainly for tests")
parser.add_argument("--oneMCfileEveryN", type=int, default=None, help="Use 1 MC file every N, where N is given by this option. Mainly for tests")
parser.add_argument("--noAuxiliaryHistograms", action="store_true", help="Remove auxiliary histograms to save memory (removed by default with --unfolding or --theoryAgnostic)")
parser.add_argument("--mtCut", type=int, default=40, help="Value for the transverse mass cut in the event selection")
parser.add_argument("--vetoGenPartPt", type=float, default=0.0, help="Minimum pT for the postFSR gen muon when defining the variation of the veto efficiency")
parser.add_argument("--noTrigger", action="store_true", help="Just for test: remove trigger HLT bit selection and trigger matching (should also remove scale factors with --noScaleFactors for it to make sense)")
#

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

isUnfolding = args.analysisMode == "unfolding"
isTheoryAgnostic = args.analysisMode in ["theoryAgnosticNormVar", "theoryAgnosticPolVar"]
isTheoryAgnosticPolVar = args.analysisMode == "theoryAgnosticPolVar"
isPoiAsNoi = (isUnfolding or isTheoryAgnostic) and args.poiAsNoi
isFloatingPOIsTheoryAgnostic = isTheoryAgnostic and not isPoiAsNoi

if isUnfolding or isTheoryAgnostic:
    parser = common.set_parser_default(parser, "excludeFlow", True)
    if isTheoryAgnostic:
        if args.genAbsYVbinEdges and any(x < 0.0 for x in args.genAbsYVbinEdges):
            raise ValueError("Option --genAbsYVbinEdges requires all positive values. Please check")
    if isFloatingPOIsTheoryAgnostic:
        logger.warning("Running theory agnostic with only nominal and mass weight histograms for now.")
        parser = common.set_parser_default(parser, "onlyMainHistograms", True)
    if isUnfolding:
        parser = common.set_parser_default(parser, "pt", [32,26.,58.])
        
# axes for W MC efficiencies with uT dependence for iso and trigger
axis_pt_eff_list = [24.,26.,28.,30.,32.,34.,36.,38.,40., 42., 44., 47., 50., 55., 60., 65.]
axis_pt_eff = hist.axis.Variable(axis_pt_eff_list, name = "pt", overflow=not args.excludeFlow, underflow=not args.excludeFlow)
if args.makeMCefficiency:
    # override the pt cuts (the binning is irrelevant since a different pt axis is used)
    nbinsPtEff = axis_pt_eff_list[-1] - axis_pt_eff_list[0]
    parser = common.set_parser_default(parser, "pt", [nbinsPtEff, axis_pt_eff_list[0], axis_pt_eff_list[-1]])

args = parser.parse_args()

thisAnalysis = ROOT.wrem.AnalysisType.Wmass

era = args.era

datasets = getDatasets(maxFiles=args.maxFiles,
                       filt=args.filterProcs,
                       excl=args.excludeProcs, 
                       nanoVersion="v9", base_path=args.dataPath, oneMCfileEveryN=args.oneMCfileEveryN,
                       extended = "msht20an3lo" not in args.pdfs,
                       era=era)

# transverse boson mass cut
mtw_min = args.mtCut

# custom template binning
template_neta = int(args.eta[0])
template_mineta = args.eta[1]
template_maxeta = args.eta[2]
logger.info(f"Eta binning: {template_neta} bins from {template_mineta} to {template_maxeta}")
template_npt = int(args.pt[0])
template_minpt = args.pt[1]
template_maxpt = args.pt[2]
logger.info(f"Pt binning: {template_npt} bins from {template_minpt} to {template_maxpt}")

# standard regular axes
axis_eta = hist.axis.Regular(template_neta, template_mineta, template_maxeta, name = "eta", overflow=False, underflow=False)
axis_pt = hist.axis.Regular(template_npt, template_minpt, template_maxpt, name = "pt", overflow=False, underflow=False)
axis_phi = hist.axis.Regular(50, -math.pi, math.pi, name = "phi", circular = True)
axis_muonJetPt = hist.axis.Regular(50, 26, 76, name = "muonJetPt", underflow=False, overflow=True)

axis_charge = common.axis_charge
axis_passIso = common.axis_passIso
axis_passMT = common.axis_passMT
axis_mt = hist.axis.Variable([0,int(mtw_min/2.),mtw_min] + list(range(mtw_min+5, 95, 5)) + [100, 120], name = "mt", underflow=False, overflow=True)
axis_met = hist.axis.Regular(25, 0., 100., name = "met", underflow=False, overflow=True)

# for mt, met, ptW plots, to compute the fakes properly (but FR pretty stable vs pt and also vs eta)
# may not exactly reproduce the same pt range as analysis, though
axis_fakes_eta = hist.axis.Regular(int((template_maxeta-template_mineta)*10/2), args.eta[1], args.eta[2], name = "eta", underflow=False, overflow=False)

axis_fakes_pt = hist.axis.Variable(common.get_binning_fakes_pt(template_minpt, template_maxpt), name = "pt", overflow=False, underflow=False)

axis_mtCat = hist.axis.Variable(common.get_binning_fakes_mt(mtw_min), name = "mt", underflow=False, overflow=True)
axes_abcd = [axis_mtCat, common.axis_relIsoCat]
axes_fakerate = [axis_fakes_eta, axis_fakes_pt, axis_charge, *axes_abcd]
columns_fakerate = ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "transverseMass", "goodMuons_relIso0"]

nominal_axes = [axis_eta, axis_pt, axis_charge, *axes_abcd]
nominal_cols = columns_fakerate

# auxiliary axes
axis_iso = hist.axis.Regular(100, 0, 25, name = "iso",underflow=False, overflow=True)
axis_relIso = hist.axis.Regular(100, 0, 1, name = "relIso",underflow=False, overflow=True)

axis_passTrigger = hist.axis.Boolean(name = "passTrigger")

axis_ut = hist.axis.Regular(40, -100, 100, overflow=True, underflow=True, name = "ut")
axes_WeffMC = [axis_eta, axis_pt_eff, axis_ut, axis_charge, axis_passIso, axis_passMT, axis_passTrigger]
# sum those groups up in post processing
groups_to_aggregate = args.aggregateGroups

if isUnfolding:
    # first and last pT bins are merged into under and overflow
    template_wpt = (template_maxpt-template_minpt)/args.genBins[0]
    min_pt_unfolding = template_minpt+template_wpt
    max_pt_unfolding = template_maxpt-template_wpt
    npt_unfolding = args.genBins[0]-2
    unfolding_axes, unfolding_cols = differential.get_pt_eta_axes(npt_unfolding, min_pt_unfolding, max_pt_unfolding, args.genBins[1] if "absEtaGen" in args.genAxes else None , flow_eta=isPoiAsNoi)
    if not isPoiAsNoi:
        datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Wmunu")
        # datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Wtaunu")

elif isTheoryAgnostic:
    theoryAgnostic_axes, theoryAgnostic_cols = differential.get_theoryAgnostic_axes(ptV_bins=args.genPtVbinEdges, absYV_bins=args.genAbsYVbinEdges, ptV_flow=isPoiAsNoi, absYV_flow=isPoiAsNoi)
    axis_helicity = helicity_utils.axis_helicity_multidim
    # the following just prepares the existence of the group for out-of-acceptance signal, but doesn't create or define the histogram yet
    if not isPoiAsNoi or (isTheoryAgnosticPolVar and args.theoryAgnosticSplitOOA): # this splitting is not needed for the normVar version of the theory agnostic
        datasets = unfolding_tools.add_out_of_acceptance(datasets, group = "Wmunu")
        groups_to_aggregate.append("WmunuOOA")

# axes for study of fakes
axis_mt_fakes = hist.axis.Regular(120, 0., 120., name = "mt", underflow=False, overflow=True)
axis_dphi_fakes = hist.axis.Regular(8, 0., np.pi, name = "DphiMuonMet", underflow=False, overflow=False)
axis_hasjet_fakes = hist.axis.Boolean(name = "hasJets") # only need case with 0 jets or > 0 for now
mTStudyForFakes_axes = [axis_eta, axis_pt, axis_charge, axis_mt_fakes, axis_passIso, axis_hasjet_fakes, axis_dphi_fakes]

## for some tests with QCD MC only
axis_eta_coarse_fakes = hist.axis.Regular(12, -2.4, 2.4, name = "eta", overflow=False, underflow=False)
axis_pt_coarse_fakes = hist.axis.Regular(8, 26, 58, name = "pt", overflow=False, underflow=False)
axis_mt_coarse_fakes = hist.axis.Regular(24, 0., 120., name = "mt", underflow=False, overflow=True)
axis_Njets_fakes = hist.axis.Regular(5, -0.5, 4.5, name = "Njets", underflow=False, overflow=True)
axis_leadjetPt_fakes = hist.axis.Regular(20, 0.0, 100.0, name = "leadjetPt", underflow=False, overflow=True)
otherStudyForFakes_axes = [axis_eta_coarse_fakes, axis_pt_coarse_fakes, axis_charge,
                           axis_mt_coarse_fakes, axis_passIso,
                           axis_Njets_fakes, axis_leadjetPt_fakes,
                           axis_dphi_fakes]


axis_met = hist.axis.Regular(100, 0., 200., name = "met", underflow=False, overflow=True)
axis_recoWpt = hist.axis.Regular(40, 0., 80., name = "recoWpt", underflow=False, overflow=True)

# define helpers
muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = wremnants.make_muon_prefiring_helpers(era = era)

qcdScaleByHelicity_helper = wremnants.theory_corrections.make_qcd_uncertainty_helper_by_helicity()

if args.noScaleFactors:
    logger.info("Running with no scale factors")
elif args.binnedScaleFactors:
    logger.info("Using binned scale factors and uncertainties")
    # add usePseudoSmoothing=True for tests with Asimov
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_binned(filename = data_dir + "/muonSF/allSmooth_GtoH3D.root", era = era, max_pt = axis_pt.edges[-1], usePseudoSmoothing=True)
else:
    logger.info("Using smoothed scale factors and uncertainties")
    muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = wremnants.make_muon_efficiency_helpers_smooth(filename = args.sfFile, era = era, 
        what_analysis=thisAnalysis, max_pt=axis_pt.edges[-1], isoEfficiencySmoothing=args.isoEfficiencySmoothing, isAltBkg=False, smooth3D=args.smooth3dsf, isoDefinition=args.isolationDefinition)
    _, muon_efficiency_helper_syst_altBkg, _ = wremnants.make_muon_efficiency_helpers_smooth(filename = args.sfFile.replace(".root","_altBkg.root"), era = era, 
        what_analysis=thisAnalysis, max_pt=axis_pt.edges[-1], isoEfficiencySmoothing=args.isoEfficiencySmoothing, isAltBkg=True, smooth3D=args.smooth3dsf, isoDefinition=args.isolationDefinition)

logger.info(f"SF file: {args.sfFile}")

pileup_helper = wremnants.make_pileup_helper(era = era)
vertex_helper = wremnants.make_vertex_helper(era = era)

calib_filepaths = common.calib_filepaths
closure_filepaths = common.closure_filepaths

diff_weights_helper = ROOT.wrem.SplinesDifferentialWeightsHelper(calib_filepaths['tflite_file']) if (args.muonScaleVariation == 'smearingWeightsSplines' or args.validationHists) else None

mc_jpsi_crctn_helper, data_jpsi_crctn_helper, jpsi_crctn_MC_unc_helper, jpsi_crctn_data_unc_helper = muon_calibration.make_jpsi_crctn_helpers(args, calib_filepaths, make_uncertainty_helper=True)

z_non_closure_parametrized_helper, z_non_closure_binned_helper = muon_calibration.make_Z_non_closure_helpers(args, calib_filepaths, closure_filepaths)

mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = muon_calibration.make_muon_calibration_helpers(args)

closure_unc_helper = wremnants.muon_calibration.make_closure_uncertainty_helper(common.closure_filepaths["parametrized"])
closure_unc_helper_A = wremnants.muon_calibration.make_uniform_closure_uncertainty_helper(0, common.correlated_variation_base_size["A"])
closure_unc_helper_M = wremnants.muon_calibration.make_uniform_closure_uncertainty_helper(2, common.correlated_variation_base_size["M"])

smearing_helper, smearing_uncertainty_helper = (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()

bias_helper = muon_calibration.make_muon_bias_helpers(args) if args.biasCalibration else None

theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
procsWithTheoryCorr = [d.name for d in datasets if d.name in common.vprocs]
if len(procsWithTheoryCorr):
    corr_helpers = theory_corrections.load_corr_helpers(procsWithTheoryCorr, theory_corrs)
else:
    corr_helpers = {}
    
# For polynominal variations
if isTheoryAgnosticPolVar:
    theoryAgnostic_helpers_minus = wremnants.helicity_utils_polvar.makehelicityWeightHelper_polvar(genVcharge=-1, fileTag=args.theoryAgnosticFileTag, filePath=args.theoryAgnosticFilePath)
    theoryAgnostic_helpers_plus  = wremnants.helicity_utils_polvar.makehelicityWeightHelper_polvar(genVcharge=1,  fileTag=args.theoryAgnosticFileTag, filePath=args.theoryAgnosticFilePath)

# recoil initialization
if not args.noRecoil:
    from wremnants import recoil_tools
    recoilHelper = recoil_tools.Recoil("highPU", args, flavor="mu")


######################################################
######################################################
######################################################
## FIXME/TODO
## next function should have been imported from theoryAgnostic_tools.py, but requires too many things as input,
## such as the helpers created here. Since it is effectively a specialization of the loop flow,
## it is part of the histmaker and is probably fine to have it here.
## In fact, having this custom function overriding the main graph is probably not the best idea, should rather use the same

# graph building for W sample with helicity weights for original theory agnostic fit with floating POIs
def setTheoryAgnosticGraph(df, results, dataset, reco_sel_GF, era, nominal_axes_thAgn, nominal_cols_thAgn, args):
    logger.info(f"Setting theory agnostic graph for {dataset.name}")

    if not args.onlyMainHistograms:
        if not args.onlyTheorySyst:
            df = syst_tools.add_L1Prefire_unc_hists(results, df, muon_prefiring_helper_stat, muon_prefiring_helper_syst, nominal_axes_thAgn, nominal_cols_thAgn, addhelicity=True)
            df = syst_tools.add_muon_efficiency_unc_hists(results, df, muon_efficiency_helper_stat, muon_efficiency_helper_syst, nominal_axes_thAgn, nominal_cols_thAgn, what_analysis=thisAnalysis, addhelicity=True)
        df = syst_tools.add_theory_hists(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, nominal_axes_thAgn, nominal_cols_thAgn, for_wmass=True, addhelicity=True)
    else:
        #FIXME: hardcoded to keep mass weights, this would be done in add_theory_hists
        df = syst_tools.define_mass_weights(df, dataset.name)
        syst_tools.add_massweights_hist(results, df, nominal_axes_thAgn, nominal_cols_thAgn, proc=dataset.name, addhelicity=True)
######################################################
######################################################
######################################################
    
smearing_weights_procs = []

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    isW = dataset.name in common.wprocs
    isWmunu = dataset.name in ["WplusmunuPostVFP", "WminusmunuPostVFP"]
    isZ = dataset.name in common.zprocs
    isZveto = isZ or dataset.name in ["DYJetsToMuMuMass10to50PostVFP"]
    isWorZ = isW or isZ
    isTop = dataset.group == "Top"
    isQCDMC = dataset.group == "QCD"
    require_prompt = "tau" not in dataset.name # for muon GEN-matching   
    storage_type=hist.storage.Double() # turn off sum weight square for systematic histograms
    
    # disable auxiliary histograms when unfolding to reduce memory consumptions, or when doing the original theory agnostic without --poiAsNoi
    auxiliary_histograms = True
    if args.noAuxiliaryHistograms or isUnfolding or isFloatingPOIsTheoryAgnostic:
        auxiliary_histograms = False

    apply_theory_corr = theory_corrs and dataset.name in corr_helpers

    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.Define("isEvenEvent", "event % 2 == 0")

    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if isUnfolding and isWmunu:
        df = unfolding_tools.define_gen_level(df, args.genLevel, dataset.name, mode="wmass")
        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = unfolding_tools.select_fiducial_space(df, mtw_min=args.mtCut, mode="wmass", accept=False)
        else:
            if not isPoiAsNoi:
                logger.debug("Select events in fiducial phase space")
                df = unfolding_tools.select_fiducial_space(df, mtw_min=args.mtCut, mode="wmass", accept=True)
                axes = [*nominal_axes, *unfolding_axes] 
                cols = [*nominal_cols, *unfolding_cols]
            
            unfolding_tools.add_xnorm_histograms(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, unfolding_axes, unfolding_cols)

    if isTheoryAgnostic and isWmunu: # should be isW to do also Wtaunu
        df = theory_tools.define_prefsr_vars(df)
        usePtOverM = False
        if isTheoryAgnosticPolVar:
            df = df.Define("qtOverQ", "ptVgen/massVgen") # FIXME: should there be a protection against mass=0 and what value to use?
            OOAthresholds = args.theoryAgnosticFileTag.split("_")
            ptVthresholdOOA   = float(OOAthresholds[0].replace("x","").replace("p","."))
            absyVthresholdOOA = float(OOAthresholds[1].replace("y","").replace("p","."))
            usePtOverM = True
        if hasattr(dataset, "out_of_acceptance"):
            logger.debug("Reject events in fiducial phase space")
            df = theoryAgnostic_tools.select_fiducial_space(df, ptVthresholdOOA, absyVthresholdOOA, accept=False, select=True, usePtOverM=usePtOverM)
        else:
            # the in-acceptance selection must usually not be used to filter signal events when doing POIs as NOIs
            if isFloatingPOIsTheoryAgnostic or (isTheoryAgnosticPolVar and args.theoryAgnosticSplitOOA):
                logger.debug("Select events in fiducial phase space for theory agnostic analysis")
                df = theoryAgnostic_tools.select_fiducial_space(df, ptVthresholdOOA, absyVthresholdOOA, accept=True, select=True, usePtOverM=usePtOverM)
                # helicity axis is special, defined through a tensor later, theoryAgnostic_ only includes W pt and rapidity for now
                if isFloatingPOIsTheoryAgnostic:
                    axes = [*nominal_axes, *theoryAgnostic_axes]
                    cols = [*nominal_cols, *theoryAgnostic_cols]
                    theoryAgnostic_tools.add_xnorm_histograms(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, theoryAgnostic_axes, theoryAgnostic_cols)

    if not args.makeMCefficiency and not args.noTrigger:
        # remove trigger, it will be part of the efficiency selection for passing trigger
        df = df.Filter(muon_selections.hlt_string(era))

    if args.halfStat:
        df = df.Filter("event % 2 == 1") # test with odd/even events

    df = muon_calibration.define_corrected_muons(df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper)

    df = muon_selections.select_veto_muons(df, nMuons=1)
    df = muon_selections.select_good_muons(df, template_minpt, template_maxpt, dataset.group, nMuons=1, use_trackerMuons=args.trackerMuons, use_isolation=False)

    # the corrected RECO muon kinematics, which is intended to be used as the nominal
    df = muon_calibration.define_corrected_reco_muon_kinematics(df)

    df = muon_selections.select_standalone_muons(df, dataset, args.trackerMuons, "goodMuons")

    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df)
    if args.makeMCefficiency:
        df = df.Define("GoodTrigObjs", f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_{era}>(TrigObj_id,TrigObj_filterBits)")
        hltString = muon_selections.hlt_string(era)
        df = df.Define("passTrigger", f"{hltString} && wrem::hasTriggerMatch(goodMuons_eta0,goodMuons_phi0,TrigObj_eta[GoodTrigObjs],TrigObj_phi[GoodTrigObjs])")
    elif not args.noTrigger:
        df = muon_selections.apply_triggermatching_muon(df, dataset, "goodMuons", era=era)

    if isWorZ:
        df = muon_validation.define_cvh_reco_muon_kinematics(df)
        reco_sel = "vetoMuonsPre"
        df = muon_calibration.define_genFiltered_recoMuonSel(df, reco_sel, require_prompt)
        reco_sel_GF = muon_calibration.getColName_genFiltered_recoMuonSel(reco_sel, require_prompt)
        df = muon_calibration.define_covMatFiltered_recoMuonSel(df, reco_sel_GF)
        df = muon_calibration.define_matched_gen_muons_covMat(df, reco_sel_GF)
        df = muon_calibration.define_matched_gen_muons_kinematics(df, reco_sel_GF)
        df = muon_calibration.calculate_matched_gen_muon_kinematics(df, reco_sel_GF)
        df = muon_calibration.define_matched_genSmeared_muon_kinematics(df, reco_sel_GF)
        df = muon_calibration.define_matched_reco_muon_kinematics(df, reco_sel_GF)

        reco_sel = "goodMuons"
        df = muon_calibration.define_matched_gen_muons_kinematics(df, reco_sel)
        df = muon_calibration.calculate_matched_gen_muon_kinematics(df, reco_sel)
        df = muon_calibration.define_matched_gen_muons_covMat(df, reco_sel)
        df = muon_calibration.define_matched_genSmeared_muon_kinematics(df, reco_sel)

        for var in ['Pt', 'Eta', 'Charge', 'Qop']:
            df = df.Define(f"{reco_sel}_{var.lower()}0_gen", f"{reco_sel}_gen{var.capitalize()}[0]")
            df = df.Define(f"{reco_sel_GF}_{var.lower()}0_gen", f"{reco_sel_GF}_gen{var.capitalize()}[0]")

            df = df.Define(f"{reco_sel}_{var.lower()}0_gen_smeared", f"{reco_sel}_genSmeared{var.capitalize()}[0]")
            df = df.Define(f"{reco_sel_GF}_{var.lower()}0_gen_smeared", f"{reco_sel_GF}_genSmeared{var.capitalize()}[0]")
            if var != 'Qop':
                df = df.Define(f"{reco_sel_GF}_{var.lower()}0_reco", f"{reco_sel_GF}_reco{var.capitalize()}[0]")
        df = df.Define(f"{reco_sel_GF}_covMat0", f"{reco_sel_GF}_covMat[0]")

        if args.validationHists:
            for reco_type in ['crctd', 'cvh', 'uncrct', 'gen_smeared']:
                df = muon_validation.define_reco_over_gen_cols(df, reco_type)

    df = df.Define("goodMuons_ptJet0", "Jet_pt[Muon_jetIdx[goodMuons][0]]")

    if args.isolationDefinition == "iso04":
        df = df.Define("goodMuons_relIso0", "Muon_pfRelIso04_all[goodMuons][0]")
        df = df.Define("goodMuons_iso0", "Muon_pfRelIso04_all[goodMuons][0]*Muon_pt[goodMuons][0]")
    elif args.isolationDefinition == "iso04vtxAgn":
        df = df.Define("goodMuons_relIso0", "Muon_vtxAgnPfRelIso04_all[goodMuons][0]")
        df = df.Define("goodMuons_iso0", "Muon_vtxAgnPfRelIso04_all[goodMuons][0]*Muon_pt[goodMuons][0]")
    elif args.isolationDefinition == "iso04chg":
        df = df.Define("goodMuons_relIso0", "Muon_pfRelIso04_chg[goodMuons][0]")
        df = df.Define("goodMuons_iso0", "Muon_pfRelIso04_chg[goodMuons][0]*Muon_pt[goodMuons][0]")
    elif args.isolationDefinition == "iso04chgvtxAgn":
        df = df.Define("goodMuons_relIso0", "Muon_vtxAgnPfRelIso04_chg[goodMuons][0]")
        df = df.Define("goodMuons_iso0", "Muon_vtxAgnPfRelIso04_chg[goodMuons][0]*Muon_pt[goodMuons][0]")
    else:
        raise NotImplementedError(f"Isolation definition {args.isolationDefinition} not implemented")

    # Jet collection actually has a pt threshold of 15 GeV in MiniAOD 
    df = df.Define("goodCleanJetsNoPt", "Jet_jetId >= 6 && (Jet_pt > 50 || Jet_puId >= 4) && abs(Jet_eta) < 2.4 && wrem::cleanJetsFromLeptons(Jet_eta,Jet_phi,Muon_correctedEta[vetoMuons],Muon_correctedPhi[vetoMuons],Electron_eta[vetoElectrons],Electron_phi[vetoElectrons])")
    df = df.Define("passIso", "goodMuons_relIso0 < 0.15")

    ########################################################################
    # define event weights here since they are needed below for some helpers
    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")            
    else:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])
        df = df.Define("weight_newMuonPrefiringSF", muon_prefiring_helper, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_correctedCharge", "Muon_looseId"])

        if era == "2016PostVFP":
            weight_expr = "weight_pu*weight_newMuonPrefiringSF*L1PreFiringWeight_ECAL_Nom"
        else:
            weight_expr = "weight_pu*L1PreFiringWeight_Muon_Nom*L1PreFiringWeight_ECAL_Nom"

        if not args.noVertexWeight:
            weight_expr += "*weight_vtx"

        # define recoil uT, muon projected on boson pt, the latter is made using preFSR variables
        # TODO: fix it for not W/Z processes
        columnsForSF = ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_SApt0", "goodMuons_SAeta0", "goodMuons_uT0", "goodMuons_charge0", "passIso"]
        df = muon_selections.define_muon_uT_variable(df, isWorZ, smooth3dsf=args.smooth3dsf, colNamePrefix="goodMuons")
        if not args.smooth3dsf:
            columnsForSF.remove("goodMuons_uT0")

        if not isQCDMC and not args.noScaleFactors:
            df = df.Define("weight_fullMuonSF_withTrackingReco", muon_efficiency_helper, columnsForSF)
            weight_expr += "*weight_fullMuonSF_withTrackingReco"

        logger.debug(f"Exp weight defined: {weight_expr}")
        df = df.Define("exp_weight", weight_expr)
        df = theory_tools.define_theory_weights_and_corrs(df, dataset.name, corr_helpers, args)

        if isWmunu and isTheoryAgnostic and not hasattr(dataset, "out_of_acceptance"):
            df = theoryAgnostic_tools.define_helicity_weights(df)

    ########################################################################

    # gen match to bare muons to select only prompt muons from MC processes, but also including tau decays
    if not dataset.is_data and not isQCDMC and not args.noGenMatchMC:
        df = theory_tools.define_postfsr_vars(df)
        df = df.Filter("wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuons],GenPart_phi[postfsrMuons],0.09)")

    if isZveto:
        df = df.Define("postfsrMuons_inAcc", f"postfsrMuons && abs(GenPart_eta) < 2.4 && GenPart_pt > {args.vetoGenPartPt}")
        df = df.Define("ZvetoCondition", "Sum(postfsrMuons_inAcc) >= 2")

    if not args.noRecoil:
        leps_uncorr = ["Muon_pt[goodMuons][0]", "Muon_eta[goodMuons][0]", "Muon_phi[goodMuons][0]", "Muon_charge[goodMuons][0]"]
        leps_corr = ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_phi0", "goodMuons_charge0"]
        df = recoilHelper.recoil_W(df, results, dataset, common.vprocs, leps_uncorr, leps_corr, cols_fakerate=columns_fakerate, axes_fakerate=axes_fakerate, mtw_min=mtw_min) # produces corrected MET as MET_corr_rec_pt/phi
    else:
        met = "DeepMETResolutionTune" if args.met == "DeepMETReso" else args.met
        df = df.Alias("MET_corr_rec_pt", f"{met}_pt")
        df = df.Alias("MET_corr_rec_phi", f"{met}_phi")

    df = df.Define("ptW", "wrem::pt_2(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)")
    df = df.Define("transverseMass", "wrem::mt_2(goodMuons_pt0, goodMuons_phi0, MET_corr_rec_pt, MET_corr_rec_phi)")

    df = df.Define("hasCleanJet", "Sum(goodCleanJetsNoPt && Jet_pt > 30) >= 1")

    df = df.Define("deltaPhiMuonMet", "std::abs(wrem::deltaPhi(goodMuons_phi0,MET_corr_rec_phi))")

    if auxiliary_histograms: 
        mTStudyForFakes = df.HistoBoost("mTStudyForFakes", mTStudyForFakes_axes, ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "transverseMass", "passIso", "hasCleanJet", "deltaPhiMuonMet", "nominal_weight"])
        results.append(mTStudyForFakes)

    # add filter of deltaPhi(muon,met) before other histograms (but after histogram mTStudyForFakes)
    if args.dphiMuonMetCut > 0.0 and not args.makeMCefficiency:
        dphiMuonMetCut = args.dphiMuonMetCut * np.pi
        df = df.Filter(f"deltaPhiMuonMet > {dphiMuonMetCut}") # pi/4 was found to be a good threshold for signal with mT > 40 GeV

    df = df.Define("passMT", f"transverseMass >= {mtw_min}")    

    if auxiliary_histograms:
        # control plots, lepton, met, to plot them later (need eta-pt to make fakes)
        results.append(df.HistoBoost("leptonPhi", [axis_phi, *axes_fakerate], ["goodMuons_phi0", *columns_fakerate, "nominal_weight"]))
        results.append(df.HistoBoost("MET", [axis_met, *axes_fakerate], ["MET_corr_rec_pt", *columns_fakerate, "nominal_weight"]))
        results.append(df.HistoBoost("METPhi", [axis_phi, *axes_fakerate], ["MET_corr_rec_phi", *columns_fakerate, "nominal_weight"]))
        results.append(df.HistoBoost("deltaPhiMuonMet", [axis_phi, *axes_fakerate], ["deltaPhiMuonMet", *columns_fakerate, "nominal_weight"]))
        results.append(df.HistoBoost("ptW", [axis_recoWpt, *axes_fakerate], ["ptW", *columns_fakerate, "nominal_weight"]))
        # for mt use axis with different binning
        results.append(df.HistoBoost("transverseMass", [axis_fakes_eta, axis_fakes_pt, axis_charge, axis_mt, common.axis_relIsoCat], ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "transverseMass", "goodMuons_relIso0", "nominal_weight"]))

        # other plots
        results.append(df.HistoBoost("iso", [axis_iso], ["goodMuons_iso0", "nominal_weight"]))
        results.append(df.HistoBoost("relIso", [axis_relIso], ["goodMuons_relIso0", "nominal_weight"]))

    if isPoiAsNoi and isW:
        if isTheoryAgnostic and isWmunu and not hasattr(dataset, "out_of_acceptance"): # TODO: might add Wtaunu at some point, not yet
            noiAsPoiHistName = Datagroups.histName("nominal", syst="yieldsTheoryAgnostic")
            logger.debug(f"Creating special histogram '{noiAsPoiHistName}' for theory agnostic to treat POIs as NOIs")
            results.append(df.HistoBoost(noiAsPoiHistName, [*nominal_axes, *theoryAgnostic_axes], [*nominal_cols, *theoryAgnostic_cols, "nominal_weight_helicity"], tensor_axes=[axis_helicity]))
            if isTheoryAgnosticPolVar:
                theoryAgnostic_helpers_cols = ["qtOverQ", "absYVgen", "chargeVgen", "csSineCosThetaPhigen", "nominal_weight"]
                # assume to have same coeffs for plus and minus (no reason for it not to be the case)
                for genVcharge in ["minus", "plus"]:
                    for coeffKey in theoryAgnostic_helpers_minus.keys():
                        logger.debug(f"Creating theory agnostic histograms with polynomial variations for {coeffKey} and {genVcharge} gen W charge")
                        helperQ = theoryAgnostic_helpers_minus[coeffKey] if genVcharge == "minus" else theoryAgnostic_helpers_plus[coeffKey]
                        df = df.Define(f"theoryAgnostic_{coeffKey}_{genVcharge}_tensor", helperQ, theoryAgnostic_helpers_cols)
                        noiAsPoiWithPolHistName = Datagroups.histName("nominal", syst=f"theoryAgnosticWithPol_{coeffKey}_{genVcharge}")
                        results.append(df.HistoBoost(noiAsPoiWithPolHistName, nominal_axes, [*nominal_cols, f"theoryAgnostic_{coeffKey}_{genVcharge}_tensor"], tensor_axes=helperQ.tensor_axes, storage=hist.storage.Double()))

        if isUnfolding:
            noiAsPoiHistName = Datagroups.histName("nominal", syst="yieldsUnfolding")
            logger.debug(f"Creating special histogram '{noiAsPoiHistName}' for unfolding to treat POIs as NOIs")
            results.append(df.HistoBoost(noiAsPoiHistName, [*nominal_axes, *unfolding_axes], [*nominal_cols, *unfolding_cols, "nominal_weight"]))       

    ## FIXME: should be isW, to include Wtaunu, but for now we only split Wmunu
    ## Note: this part is only for the original theory agnostic with fully floating POIs
    elif isWmunu and isFloatingPOIsTheoryAgnostic and not hasattr(dataset, "out_of_acceptance"):
        results.append(df.HistoBoost("nominal", axes, [*cols, "nominal_weight_helicity"], tensor_axes=[axis_helicity]))
        setTheoryAgnosticGraph(df, results, dataset, reco_sel_GF, era, axes, cols, args)
        # End graph here only for standard theory agnostic analysis, otherwise use same loop as traditional analysis
        return results, weightsum

    if not args.onlyMainHistograms:
        syst_tools.add_QCDbkg_jetPt_hist(results, df, axes, cols, jet_pt=30, storage_type=storage_type)

    if dataset.is_data:
        nominal = df.HistoBoost("nominal", axes, cols)
        results.append(nominal)
    else:  
        nominal = df.HistoBoost("nominal", axes, [*cols, "nominal_weight"])
        results.append(nominal)
        results.append(df.HistoBoost("nominal_weight", [hist.axis.Regular(200, -4, 4)], ["nominal_weight"], storage=hist.storage.Double()))

        if args.makeMCefficiency:
            cols_WeffMC = ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_uT0", "goodMuons_charge0",
                           "passIso", "passMT", "passTrigger"]
            yieldsForWeffMC = df.HistoBoost("yieldsForWeffMC", axes_WeffMC, [*cols_WeffMC, "nominal_weight"])
            results.append(yieldsForWeffMC)

        if not args.noRecoil and args.recoilUnc:
            df = recoilHelper.add_recoil_unc_W(df, results, dataset, cols, axes, "nominal", storage_type=storage_type)
        if apply_theory_corr:
            results.extend(theory_tools.make_theory_corr_hists(df, "nominal", axes, cols, corr_helpers[dataset.name], theory_corrs, 
                modify_central_weight=not args.theoryCorrAltOnly, isW = isW, storage_type=storage_type)
            )
        if isWorZ:
            cols_gen, cols_gen_smeared = muon_calibration.make_alt_reco_and_gen_hists(df, results, axes, cols, reco_sel_GF)
            if args.validationHists: 
                muon_validation.make_reco_over_gen_hists(df, results)

    if not dataset.is_data and not args.onlyMainHistograms:

        if not args.onlyTheorySyst:
            if not isQCDMC and not args.noScaleFactors:
                df = syst_tools.add_muon_efficiency_unc_hists(results, df, muon_efficiency_helper_stat, muon_efficiency_helper_syst, axes, cols, 
                    what_analysis=thisAnalysis, smooth3D=args.smooth3dsf, storage_type=storage_type)
                df = syst_tools.add_muon_efficiency_unc_hists_altBkg(results, df, muon_efficiency_helper_syst_altBkg, axes, cols, 
                    what_analysis=thisAnalysis, smooth3D=args.smooth3dsf, storage_type=storage_type)
            df = syst_tools.add_L1Prefire_unc_hists(results, df, muon_prefiring_helper_stat, muon_prefiring_helper_syst, axes, cols, storage_type=storage_type)
            # luminosity, as shape variation despite being a flat scaling to facilitate propagation to fakes
            df = syst_tools.add_luminosity_unc_hists(results, df, args, axes, cols, storage_type=storage_type)
            if isZveto:
                df = syst_tools.add_scaledByCondition_unc_hists(results, df, args, axes, cols, "weight_ZmuonVeto", "ZmuonVeto", "ZvetoCondition", 2.0, storage_type=storage_type)

        # n.b. this is the W analysis so mass weights shouldn't be propagated
        # on the Z samples (but can still use it for dummy muon scale)

        if isWorZ:

            df = syst_tools.add_theory_hists(results, df, args, dataset.name, corr_helpers, qcdScaleByHelicity_helper, axes, cols, for_wmass=True, storage_type=storage_type)

            # Don't think it makes sense to apply the mass weights to scale leptons from tau decays
            if not args.onlyTheorySyst and not "tau" in dataset.name:
                df = syst_tools.add_muonscale_hist(results, df, args.muonCorrEtaBins, args.muonCorrMag, isW, axes, cols, storage_type=storage_type)
                if args.muonScaleVariation == 'smearingWeightsGaus':
                    df = syst_tools.add_muonscale_smeared_hist(results, df, args.muonCorrEtaBins, args.muonCorrMag, isW, axes, cols_gen_smeared, storage_type=storage_type)

            ####################################################
            # nuisances from the muon momemtum scale calibration 
            if (args.muonCorrData in ["massfit", "lbl_massfit"]):
                input_kinematics = [
                    f"{reco_sel_GF}_recoPt",
                    f"{reco_sel_GF}_recoEta",
                    f"{reco_sel_GF}_recoCharge",
                    f"{reco_sel_GF}_genPt",
                    f"{reco_sel_GF}_genEta",
                    f"{reco_sel_GF}_genCharge"
                ]
                if diff_weights_helper:
                    df = df.Define(f'{reco_sel_GF}_response_weight', diff_weights_helper, [*input_kinematics])
                    input_kinematics.append(f'{reco_sel_GF}_response_weight')

                # muon scale variation from stats. uncertainty on the jpsi massfit
                df = muon_calibration.add_jpsi_crctn_stats_unc_hists(
                    args, df, axes, results, cols, cols_gen_smeared,
                    calib_filepaths, jpsi_crctn_data_unc_helper, smearing_weights_procs,
                    reco_sel_GF, dataset.name, isW, storage_type=storage_type
                )
                # add the ad-hoc Z non-closure nuisances from the jpsi massfit to muon scale unc
                df = muon_calibration.add_jpsi_crctn_Z_non_closure_hists(
                    args, df, axes, results, cols, cols_gen_smeared,
                    z_non_closure_parametrized_helper, z_non_closure_binned_helper, reco_sel_GF, storage_type=storage_type
                )
                # add nuisances from the data/MC resolution mismatch
                df = muon_calibration.add_resolution_uncertainty(df, axes, results, cols, smearing_uncertainty_helper, reco_sel_GF, storage_type=storage_type)
                if args.validationHists:
                    df = muon_validation.make_hists_for_muon_scale_var_weights(
                        df, axes, results, cols, cols_gen_smeared
                    )

                # extra uncertainties from non-closure stats
                df = df.Define("muonScaleClosSyst_responseWeights_tensor_splines", closure_unc_helper,
                    [*input_kinematics, "nominal_weight"]
                )
                nominal_muonScaleClosSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosSyst_responseWeights", axes,
                    [*cols, "muonScaleClosSyst_responseWeights_tensor_splines"],
                    tensor_axes = closure_unc_helper.tensor_axes,
                    storage = hist.storage.Double()
                )
                results.append(nominal_muonScaleClosSyst_responseWeights)

                # extra uncertainties for A (fully correlated)
                df = df.Define("muonScaleClosASyst_responseWeights_tensor_splines", closure_unc_helper_A,
                    [*input_kinematics, "nominal_weight"]
                )
                nominal_muonScaleClosASyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosASyst_responseWeights", axes,
                    [*cols, "muonScaleClosASyst_responseWeights_tensor_splines"],
                    tensor_axes = closure_unc_helper_A.tensor_axes,
                    storage = hist.storage.Double()
                )
                results.append(nominal_muonScaleClosASyst_responseWeights)

                # extra uncertainties for M (fully correlated)
                df = df.Define("muonScaleClosMSyst_responseWeights_tensor_splines", closure_unc_helper_M,
                    [*input_kinematics, "nominal_weight"]
                )
                nominal_muonScaleClosMSyst_responseWeights = df.HistoBoost(
                    "nominal_muonScaleClosMSyst_responseWeights", axes,
                    [*cols, "muonScaleClosMSyst_responseWeights_tensor_splines"],
                    tensor_axes = closure_unc_helper_M.tensor_axes,
                    storage = hist.storage.Double()
                )
                results.append(nominal_muonScaleClosMSyst_responseWeights)

            ####################################################

            df = df.Define("Muon_cvhMomCov", "wrem::splitNestedRVec(Muon_cvhMomCov_Vals, Muon_cvhMomCov_Counts)")

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        if len(smearing_weights_procs) > 0 and smearing_weights_procs[-1] == dataset.name:
            smearing_weights_procs[-1] = dataset.name+"OOA"
        dataset.name = dataset.name+"OOA"

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)
if not args.onlyMainHistograms and args.muonScaleVariation == 'smearingWeightsGaus' and not isFloatingPOIsTheoryAgnostic:
    logger.debug("Apply smearingWeights")
    muon_calibration.transport_smearing_weights_to_reco(
        resultdict,
        smearing_weights_procs,
        nonClosureScheme = args.nonClosureScheme
    )
if args.validationHists:
    muon_validation.muon_scale_variation_from_manual_shift(resultdict)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, groups_to_aggregate)

output_tools.write_analysis_output(resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args)
