#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "Model.h"
#include "StringEditModel.h"
#include "WordAlignmentFeatureGen.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <math.h>
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
  const int iDel = alphabet->lookup("0_S:Del1");
  BOOST_REQUIRE(iDel >= 0);
  W.add(iDel, -100);  
  const int iIns = alphabet->lookup("0_S:Ins1");
  BOOST_REQUIRE(iIns >= 0);
  W.add(iIns, -100);
  const int iSub = alphabet->lookup("0_S:Sub11");
  BOOST_REQUIRE(iSub >= 0);
  W.add(iSub, -100);
  
  // Check that the total mass is correct.
  LogWeight totalMass = model->totalMass(W, *pair, label);
  BOOST_CHECK_CLOSE((double)totalMass, -300, 1e-8);
  
  FeatureVector<LogWeight> fv(d, true);
  BOOST_REQUIRE(!fv.isDense());
  LogWeight totalMassAlt = model->expectedFeatures(W, fv, *pair, label, false);
  BOOST_CHECK_CLOSE((double)totalMass, (double)totalMassAlt, 1e-8);
  
  const int iMat = alphabet->lookup("0_S:Mat11");
  BOOST_REQUIRE(iMat >= 0);
  const int iBias = alphabet->lookup("0_Bias");
  BOOST_REQUIRE(iBias >= 0);
  
  // Check that the (unnormalized) expected value of each feature is correct.  
  BOOST_CHECK_CLOSE((double)fv.getValueAtIndex(iIns), -298.9014, 1e-4);
  BOOST_CHECK_CLOSE((double)fv.getValueAtIndex(iDel), -497.9206, 1e-4);
  BOOST_CHECK_CLOSE((double)fv.getValueAtIndex(iSub), -398.2082, 1e-4);
  BOOST_CHECK_CLOSE((double)fv.getValueAtIndex(iMat), -298.2082, 1e-4);
  BOOST_CHECK_CLOSE((double)fv.getValueAtIndex(iBias), -300.0000, 1e-4);

  // Check that the (normalized) expected value of each feature is correct.
  fv.timesEquals(-totalMass);
  BOOST_CHECK_CLOSE(exp(fv.getValueAtIndex(iIns)), 3, 1e-4);
  BOOST_CHECK_SMALL(exp(fv.getValueAtIndex(iDel)), 1e-4);
  BOOST_CHECK_SMALL(exp(fv.getValueAtIndex(iSub)), 1e-4);
  BOOST_CHECK_CLOSE(exp(fv.getValueAtIndex(iMat)), 6, 1e-4);
  BOOST_CHECK_CLOSE(exp(fv.getValueAtIndex(iBias)), 1, 1e-4);
  
  // Check that the Viterbi score is correct.
  RealWeight viterbiScore = model->viterbiScore(W, *pair, label);
  BOOST_CHECK_CLOSE((double)viterbiScore, -300, 1e-8);
  
  FeatureVector<RealWeight> realFv(d, true);
  BOOST_REQUIRE(!realFv.isDense());
  realFv.zero();
  RealWeight maxScore = model->maxFeatures(W, realFv, *pair, label, true);
  BOOST_CHECK_CLOSE((double)maxScore, (double)viterbiScore, 1e-8);
  
  // Check that values in the max-scoring feature vector are correct.  
  BOOST_CHECK_CLOSE((double)realFv.getValueAtIndex(iIns), 3, 1e-4);
  BOOST_CHECK_SMALL((double)realFv.getValueAtIndex(iDel), 1e-4);
  BOOST_CHECK_SMALL((double)realFv.getValueAtIndex(iSub), 1e-4);
  BOOST_CHECK_CLOSE((double)realFv.getValueAtIndex(iMat), 6, 1e-4);
  BOOST_CHECK_CLOSE((double)realFv.getValueAtIndex(iBias), 1, 1e-4);
}
