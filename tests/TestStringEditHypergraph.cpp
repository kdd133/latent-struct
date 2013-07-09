#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "Model.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "WordAlignmentFeatureGen.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using namespace boost;

BOOST_AUTO_TEST_CASE(testStringEditHypergraph)
{
  const int argc = 9;
  char* argv[argc];
  size_t i = 0;
  argv[i++] = (char*) "latent_struct";
  argv[i++] = (char*) "--order=1";
  argv[i++] = (char*) "--no-align-ngrams";
  argv[i++] = (char*) "--no-collapsed-align-ngrams";
  argv[i++] = (char*) "--state-unigrams-only";
  argv[i++] = (char*) "--no-normalize";
  argv[i++] = (char*) "--bias-no-normalize";
  argv[i++] = (char*) "--no-final-arc-feats";
  argv[i++] = (char*) "--exact-match-state";
  
  shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
  shared_ptr<BiasFeatureGen> fgenObs(new BiasFeatureGen(alphabet));
  int ret = fgenObs->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  shared_ptr<WordAlignmentFeatureGen> fgenLat(new WordAlignmentFeatureGen(
      alphabet));
  ret = fgenLat->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  Model* model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
  ret = model->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  
  std::vector<Model*> models;
  models.push_back(model);
  
  StringPair* pair = new StringPair("stress", "sutoressu");
  pair->setId(0);
  // Since our "dataset" will have only one example (and therefore only one
  // unique label), TrainingObjective.gatherFeatures() requires the label to
  // be zero, even though this would actually be a positive example.
  Label label = 0;
  Example example(pair, label);
  Dataset data;
  data.addExample(example);
  
  LogLinearMulti objective(data, models);
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective.gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int d = alphabet->size();
  BOOST_REQUIRE_EQUAL(d, 5);
  
  // Set the weights of Del, Ins, and Sub to -100; implicitly leave Mat to be 0.
  WeightVector W(d);  
  const int iDel = alphabet->lookup("S:Del1", 0, false);
  BOOST_REQUIRE(iDel >= 0);
  W.add(iDel, -100);  
  const int iIns = alphabet->lookup("S:Ins1", 0, false);
  BOOST_REQUIRE(iIns >= 0);
  W.add(iIns, -100);
  const int iSub = alphabet->lookup("S:Sub11", 0, false);
  BOOST_REQUIRE(iSub >= 0);
  W.add(iSub, -100);
  
  // Check that the total mass is correct.
  LogWeight totalMass = model->totalMass(W, *pair, label);
  BOOST_CHECK_CLOSE((double)totalMass, -300, 1e-8);
  
  SparseLogVec fv(d);
  LogWeight totalMassAlt = model->expectedFeatures(W, &fv, *pair, label, false);
  BOOST_CHECK_CLOSE((double)totalMass, (double)totalMassAlt, 1e-8);
  
  const int iMat = alphabet->lookup("S:Mat11", 0, false);
  BOOST_REQUIRE(iMat >= 0);
  const int iBias = alphabet->lookup("Bias", 0, false);
  BOOST_REQUIRE(iBias >= 0);
  
  // Check that the (unnormalized) expected value of each feature is correct.  
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iIns]), -298.9014, 1e-4);
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iDel]), -497.9206, 1e-4);
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iSub]), -398.2082, 1e-4);
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iMat]), -298.2082, 1e-4);
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iBias]), -300.0000, 1e-4);

  // Check that the (normalized) expected value of each feature is correct.
  fv /= totalMass;
  BOOST_CHECK_CLOSE(exp(((LogWeight)fv[iIns])), 3, 1e-4);
  BOOST_CHECK_SMALL(exp(((LogWeight)fv[iDel])), 1e-4);
  BOOST_CHECK_SMALL(exp(((LogWeight)fv[iSub])), 1e-4);
  BOOST_CHECK_CLOSE(exp(((LogWeight)fv[iMat])), 6, 1e-4);
  BOOST_CHECK_CLOSE(exp(((LogWeight)fv[iBias])), 1, 1e-4);
  
  // Check that the Viterbi score is correct.
  double viterbiScore = model->viterbiScore(W, *pair, label);
  BOOST_CHECK_CLOSE(viterbiScore, -300, 1e-8);
  
  SparseRealVec realFv(d);
  double maxScore = model->maxFeatures(W, &realFv, *pair, label, true);
  BOOST_CHECK_CLOSE(maxScore, viterbiScore, 1e-8);
  
  // Check that values in the max-scoring feature vector are correct.  
  BOOST_CHECK_CLOSE((double)realFv[iIns], 3, 1e-4);
  BOOST_CHECK_SMALL((double)realFv[iDel], 1e-4);
  BOOST_CHECK_SMALL((double)realFv[iSub], 1e-4);
  BOOST_CHECK_CLOSE((double)realFv[iMat], 6, 1e-4);
  BOOST_CHECK_CLOSE((double)realFv[iBias], 1, 1e-4);
}
