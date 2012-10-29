#define BOOST_TEST_DYN_LINK

#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LogLinearMulti.h"
#include "Model.h"
#include "RealWeight.h"
#include "StringEditModel.h"
#include "WordAlignmentFeatureGen.h"
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>
using namespace boost;

/*
 * Build a first-order edit distance transducer, but only fire zero-order
 * features. Run the Viterbi algorithm to compute the weight of the max-scoring
 * path through the graph, as well as the feature counts from this path.
 * The reason we build the graph in this manner is to duplicate the behaviour
 * of an existing implementation whose dynamic programming routines have
 * been verified to be correct. Note that we can change --order=1 to --order=0
 * below and still get the same result.
 */
BOOST_AUTO_TEST_CASE(testStringEditViterbi)
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
  Model* model = new StringEditModel<StdFeatArc>(fgenLat, fgenObs);
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
  
  // Check that the Viterbi score is correct.
  RealWeight viterbiScore = model->viterbiScore(W, *pair, label);
  BOOST_CHECK_CLOSE(viterbiScore.value(), -300, 1e-8);
  
  FeatureVector<RealWeight> fv(d, true);
  BOOST_REQUIRE(!fv.isDense());
  RealWeight maxScore = model->maxFeatures(W, fv, *pair, label, true);
  BOOST_CHECK_CLOSE(maxScore.value(), viterbiScore.value(), 1e-8);
  
  const int iMat = alphabet->lookup("0_S:Mat11");
  BOOST_REQUIRE(iMat >= 0);
  const int iBias = alphabet->lookup("0_Bias");
  BOOST_REQUIRE(iBias >= 0);
  
  // Check that values in the max-scoring feature vector are correct.  
  BOOST_CHECK_CLOSE(fv.getValueAtIndex(iIns).value(), 3, 1e-4);
  BOOST_CHECK_SMALL(fv.getValueAtIndex(iDel).value(), 1e-4);
  BOOST_CHECK_SMALL(fv.getValueAtIndex(iSub).value(), 1e-4);
  BOOST_CHECK_CLOSE(fv.getValueAtIndex(iMat).value(), 6, 1e-4);
  BOOST_CHECK_CLOSE(fv.getValueAtIndex(iBias).value(), 1, 1e-4);
}
