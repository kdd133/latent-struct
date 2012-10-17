#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE UnitTests

#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "Model.h"
#include "StringEditModel.h"
#include "WordAlignmentFeatureGen.h"
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <math.h>
#include <vector>
using namespace boost;

/*
 * Build a first-order edit distance transducer, but only fire zero-order
 * features. Run the forward-backward algorithm to compute the total weight of
 * all paths through the graph, as well as the expectation over features.
 * The reason we build the graph in this manner is to duplicate the behaviour
 * of an existing implementation whose dynamic programming routines have
 * been verified to be correct. Note that we can change --order=1 to --order=0
 * below and still get the same result.
 */
BOOST_AUTO_TEST_CASE(testStringEdit)
{
  const int argc = 8;
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
  
  shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
  shared_ptr<BiasFeatureGen> fgenObs(new BiasFeatureGen(alphabet));
  int ret = fgenObs->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  shared_ptr<WordAlignmentFeatureGen> fgenLat(new WordAlignmentFeatureGen(
      alphabet));
  ret = fgenLat->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  Model* model = new StringEditModel<LogFeatArc>(fgenLat, fgenObs);
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
  BOOST_REQUIRE_EQUAL(d, 4);
  
  // Set the weights of Del and Ins to -100; implicitly leave Rep to be zero.
  WeightVector W(d);  
  const int iDel = alphabet->lookup("0_S:Del1");
  BOOST_REQUIRE(iDel >= 0);
  W.add(iDel, -100);  
  const int iIns = alphabet->lookup("0_S:Ins1");
  BOOST_REQUIRE(iIns >= 0);
  W.add(iIns, -100);
  
  // Check that the total mass is correct.
  LogWeight totalMass = model->totalMass(W, *pair, label);
  BOOST_CHECK_CLOSE(totalMass.value(), -295.5691832011567, 1e-8);
  
  FeatureVector<LogWeight> fv(d, true);
  BOOST_REQUIRE(!fv.isDense());
  LogWeight totalMassAlt = model->expectedFeatures(W, fv, *pair, label, false);
  BOOST_CHECK_CLOSE(totalMass.value(), totalMassAlt.value(), 1e-8);
  
  const int iRep = alphabet->lookup("0_S:Rep11");
  BOOST_REQUIRE(iRep >= 0);
  const int iBias = alphabet->lookup("0_Bias");
  BOOST_REQUIRE(iBias >= 0);
  
  // Check that the (unnormalized) expected value of each feature is correct.  
  BOOST_CHECK_CLOSE(fv.getValueAtLocation(iIns).value(), -294.4706, 1e-4);
  BOOST_CHECK_CLOSE(fv.getValueAtLocation(iDel).value(), -493.3720, 1e-4);
  BOOST_CHECK_CLOSE(fv.getValueAtLocation(iRep).value(), -293.7774, 1e-4);
  BOOST_CHECK_CLOSE(fv.getValueAtLocation(iBias).value(), -295.5692, 1e-4);

  // Check that the (normalized) expected value of each feature is correct.
  fv.timesEquals(-totalMass);
  BOOST_CHECK_CLOSE(exp(fv.getValueAtLocation(iIns).value()), 3, 1e-4);
  BOOST_CHECK_SMALL(exp(fv.getValueAtLocation(iDel).value()), 1e-4);
  BOOST_CHECK_CLOSE(exp(fv.getValueAtLocation(iRep).value()), 6, 1e-4);
  BOOST_CHECK_CLOSE(exp(fv.getValueAtLocation(iBias).value()), 1, 1e-4);
}
