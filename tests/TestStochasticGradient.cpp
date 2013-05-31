#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "StochasticGradientOptimizer.h"
#include "LogLinearBinary.h"
#include "Parameters.h"
#include "RegularizerL2.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "Utility.h"
#include "WordAlignmentFeatureGen.h"
#include "WordPairReader.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost;

BOOST_AUTO_TEST_CASE(testStochasticGradient)
{
  const int argc = 10;
  char* argv[argc];
  argv[0] = (char*) "latent_struct";
  argv[1] = (char*) "--order=0";
  argv[2] = (char*) "--no-align-ngrams";
  argv[3] = (char*) "--no-collapsed-align-ngrams";
  argv[4] = (char*) "--restarts=2";
  argv[5] = (char*) "--quietX";
  argv[6] = (char*) "--no-normalize";
  argv[7] = (char*) "--bias-no-normalize";
  argv[8] = (char*) "--no-final-arc-feats";
  argv[9] = (char*) "--validation=0.4";
  
  shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
  shared_ptr<BiasFeatureGen> fgenObs(new BiasFeatureGen(alphabet));
  int ret = fgenObs->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  shared_ptr<WordAlignmentFeatureGen> fgenLat(new WordAlignmentFeatureGen(
      alphabet));
  ret = fgenLat->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  Model* model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
  ret = model->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  
  std::vector<Model*> models;
  models.push_back(model);  
  
  Dataset trainData;
  WordPairReader reader;
  bool badFile = Utility::loadDataset(reader, "he_tiny", trainData);
  BOOST_REQUIRE(!badFile);
  
  shared_ptr<TrainingObjective> objective(new LogLinearBinary(trainData,
      models));
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective->gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int d = alphabet->size();
  BOOST_REQUIRE_EQUAL(d, 4);
  
  Parameters W = objective->getDefaultParameters(d);
  
  // set the feature weight for bias class y=1 to one
  int index = alphabet->lookup("Bias", 1, false);
  BOOST_REQUIRE(index >= 0);
  W.add(index, 1.0);
  BOOST_REQUIRE_EQUAL(W[index], 1.0);
  
  RealVec gradFv(d);
  double fval;
  objective->valueAndGradient(W, fval, gradFv);
  BOOST_CHECK_CLOSE(0.81326168751, fval, 1e-8);
  BOOST_CHECK_CLOSE(0.23105857863, gradFv[0], 1e-8);
  BOOST_CHECK_CLOSE(1.12714370334, gradFv[1], 1e-8);
  BOOST_CHECK_CLOSE(0.91161441403, gradFv[2], 1e-8);
  BOOST_CHECK_CLOSE(0.39367847911, gradFv[3], 1e-8);
  
  const double beta = 0.1;
  shared_ptr<Regularizer> l2(new RegularizerL2(beta));
  BOOST_REQUIRE_EQUAL(l2->getBeta(), beta);
  StochasticGradientOptimizer opt(objective, l2);
  ret = opt.processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret); 
  
  double fvalOpt = 0.0;
  Optimizer::status status = opt.train(W, fvalOpt, 1e-5);
  BOOST_REQUIRE(status == Optimizer::CONVERGED);
  
  objective->valueAndGradient(W, fval, gradFv);
  l2->addRegularization(W, fval, gradFv);
  BOOST_CHECK_CLOSE(0.67917812871, fval, 1e-8);
}
