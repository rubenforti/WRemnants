import ROOT
from wremnants import muon_calibration
from wremnants import theory_tools
from utilities import common, logging

logger = logging.child_logger(__name__)

def make_QCD_MC_test_histograms(df, results):

    df = df.Define("goodMuons_hasJet0", "Muon_jetIdx[goodMuons][0] != -1 ")
    df = df.Define("goodMuons_jetpt0", "goodMuons_hasJet0 ? Jet_pt[Muon_jetIdx[goodMuons][0]] : goodMuons_pt0")
    df = df.Define("goodMuons_genPartFlav0", "Muon_genPartFlav[goodMuons][0]")
    df = df.Define("nJetsClean", "Sum(goodCleanJetsNoPt)")
    df = df.Define("leadjetPt", "(nJetsClean > 0) ? Jet_pt[goodCleanJetsNoPt][0] : 0.0")
    #
    results.append(df.HistoBoost("Muon_genPartFlav", [hist.axis.Regular(6,-0.5,5.5,name="Muon genPart flavor")], ["goodMuons_genPartFlav0", "nominal_weight"]))
    #
    otherStudyForFakes = df.HistoBoost("otherStudyForFakes", otherStudyForFakes_axes, ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "transverseMass", "passIso", "nJetsClean", "leadjetPt", "deltaPhiMuonMet", "nominal_weight"])
    results.append(otherStudyForFakes)
    # gen match studies
    df = df.Define("postfsrMuonsStatus1", "GenPart_status == 1 && abs(GenPart_pdgId) == 13")
    df = df.Define("postfsrMuonsStatus1prompt", "postfsrMuonsStatus1 && (GenPart_statusFlags & 1 || GenPart_statusFlags & (1 << 5))")
    df = df.Define("postfsrMuonsStatus1notPrompt", "postfsrMuonsStatus1 && !(GenPart_statusFlags & 1 || GenPart_statusFlags & (1 << 5))")
    #
    df = df.Define("muonGenMatchStatus1", "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuonsStatus1],GenPart_phi[postfsrMuonsStatus1])")
    df = df.Define("muonGenMatchStatus1prompt", "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuonsStatus1prompt],GenPart_phi[postfsrMuonsStatus1prompt])")
    df = df.Define("muonGenMatchStatus1notPrompt", "wrem::hasMatchDR2(goodMuons_eta0,goodMuons_phi0,GenPart_eta[postfsrMuonsStatus1notPrompt],GenPart_phi[postfsrMuonsStatus1notPrompt])")
    #
    axis_match = hist.axis.Boolean(name = "hasMatch")
    etaPtGenMatchStatus1 = df.HistoBoost("etaPtGenMatchStatus1", [axis_eta, axis_pt, axis_match], ["goodMuons_eta0", "goodMuons_pt0", "muonGenMatchStatus1", "nominal_weight"])
    results.append(etaPtGenMatchStatus1)
    etaPtGenMatchStatus1prompt = df.HistoBoost("etaPtGenMatchStatus1prompt", [axis_eta, axis_pt, axis_match], ["goodMuons_eta0", "goodMuons_pt0", "muonGenMatchStatus1prompt", "nominal_weight"])
    results.append(etaPtGenMatchStatus1prompt)
    etaPtGenMatchStatus1notPrompt = df.HistoBoost("etaPtGenMatchStatus1notPrompt", [axis_eta, axis_pt, axis_match], ["goodMuons_eta0", "goodMuons_pt0", "muonGenMatchStatus1notPrompt", "nominal_weight"])
    results.append(etaPtGenMatchStatus1notPrompt)
    ### gen mT and reco mT
    df = df.Define("transverseMass_genMetRecoMuon", "wrem::mt_2(goodMuons_pt0, goodMuons_phi0, GenMET_pt, GenMET_phi)")
    axis_genmt = hist.axis.Regular(120, 0., 120., name = "genMt", underflow=False, overflow=True)
    axis_recomet = hist.axis.Regular(120, 0., 120., name = "recoMet", underflow=False, overflow=True)
    axis_genmet = hist.axis.Regular(120, 0., 120., name = "genMet", underflow=False, overflow=True)
    etaPtMtGenMt = df.HistoBoost("etaPtMtGenMt", [axis_eta, axis_pt, axis_mt_fakes, axis_genmt, axis_passIso], ["goodMuons_eta0", "goodMuons_pt0", "transverseMass", "transverseMass_genMetRecoMuon", "passIso", "nominal_weight"])
    results.append(etaPtMtGenMt)
    etaPtMetGenMet = df.HistoBoost("etaPtMetGenMet", [axis_eta, axis_pt, axis_recomet, axis_genmet, axis_passIso], ["goodMuons_eta0", "goodMuons_pt0", "MET_corr_rec_pt", "GenMET_pt", "passIso", "nominal_weight"])
    results.append(etaPtMetGenMet)
    #
    muonHasJet = df.HistoBoost("muonHasJet", [hist.axis.Regular(2,-0.5,1.5,name="hasJet"), hist.axis.Regular(2,-0.5,1.5,name="passIsolation")], ["goodMuons_hasJet0", "passIso", "nominal_weight"])
    results.append(muonHasJet)
    # add a cut to a new branch of the dataframe
    dfalt = df.Filter("goodMuons_hasJet0")
    axis_jetpt = hist.axis.Regular(40,20,60, name = "jetpt", overflow=False, underflow=False)
    axis_muonpt = hist.axis.Regular(40,20,60, name = "pt", overflow=False, underflow=False)
    muonPtVsJetPt = dfalt.HistoBoost("muonPtVsJetPt", [axis_muonpt, axis_jetpt, axis_passIso], ["goodMuons_pt0", "goodMuons_jetpt0", "passIso", "nominal_weight"])
    results.append(muonPtVsJetPt)

    return df
