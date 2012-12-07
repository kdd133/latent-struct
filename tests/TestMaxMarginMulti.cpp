#define BOOST_TEST_DYN_LINK

#include "AlignmentTransducer.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "BmrmOptimizer.h"
#include "Dataset.h"
#include "EmOptimizer.h"
#include "MaxMarginMulti.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "Utility.h"
#include "WordAlignmentFeatureGen.h"
#include "WordPairReader.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost;


BOOST_AUTO_TEST_CASE(testMaxMarginMulti)
{
  const int argc = 9;
  char* argv[argc];
  argv[0] = (char*) "latent_struct";
  argv[1] = (char*) "--order=0";
  argv[2] = (char*) "--no-align-ngrams";
  argv[3] = (char*) "--no-collapsed-align-ngrams";
  argv[4] = (char*) "--restarts=2";
  argv[5] = (char*) "--quiet";
  argv[6] = (char*) "--no-normalize";
  argv[7] = (char*) "--bias-no-normalize";
  argv[8] = (char*) "--no-final-arc-feats";
  
  shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
  shared_ptr<BiasFeatureGen> fgenObs(new BiasFeatureGen(alphabet));
  int ret = fgenObs->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  shared_ptr<WordAlignmentFeatureGen> fgenLat(new WordAlignmentFeatureGen(
      alphabet));
  ret = fgenLat->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  Model* model = new StringEditModel<AlignmentTransducer<StdFeatArc> >(fgenLat,
      fgenObs);
  ret = model->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  
  vector<Model*> models;
  models.push_back(model);  
  
  Dataset trainData;
  WordPairReader reader;
  bool badFile = Utility::loadDataset(reader, "he_tiny", trainData);
  BOOST_REQUIRE(!badFile);
  
  MaxMarginMulti objective(trainData, models);
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective.gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int d = alphabet->size();
  BOOST_REQUIRE_EQUAL(d, 8);
  
  WeightVector W(d);
  
  // set the feature weight for bias class y=0 to one
  int index = alphabet->lookup("0_Bias", false);
  BOOST_REQUIRE(index >= 0);
  W.add(index, 1.0);
  BOOST_REQUIRE_EQUAL(W.getWeight(index), 1.0);
  
  RealVec gradFv(d);
  double fval;
  objective.valueAndGradient(W, fval, gradFv);
  
  BOOST_CHECK_CLOSE(1, fval, 1e-8);
  BOOST_CHECK_CLOSE(0.50, gradFv[0], 1e-8);
  BOOST_CHECK_CLOSE(0.85, gradFv[1], 1e-8);
  BOOST_CHECK_CLOSE(0.35, gradFv[2], 1e-8);
  BOOST_CHECK_CLOSE(4.65, gradFv[3], 1e-8);
  BOOST_CHECK_CLOSE(-0.50, gradFv[4], 1e-8);
  BOOST_CHECK_CLOSE(0, gradFv[5], 1e-8);
  BOOST_CHECK_CLOSE(0, gradFv[6], 1e-8);
  BOOST_CHECK_CLOSE(0, gradFv[7], 1e-8);
  
  shared_ptr<Optimizer> convexOpt(new BmrmOptimizer(objective));
  ret = convexOpt->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret);
  EmOptimizer opt(objective, convexOpt);
  ret = opt.processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret);
  opt.setBeta(0.1);
  BOOST_REQUIRE_EQUAL(opt.getBeta(), 0.1);
  
  double fvalOpt = 0.0;
  Optimizer::status status = opt.train(W, fvalOpt, 1e-5);
  BOOST_REQUIRE(status == Optimizer::CONVERGED);
  
  objective.valueAndGradient(W, fval, gradFv);
  Utility::addRegularizationL2(W, opt.getBeta(), fval, gradFv);
  BOOST_CHECK_CLOSE(0.60818096163, fval, 1e-8);
}
