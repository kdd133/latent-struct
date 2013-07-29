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
  argv[2] = (char*) "--no-collapsed-align-ngrams";
  argv[3] = (char*) "--quiet";
  argv[4] = (char*) "--no-normalize";
  argv[5] = (char*) "--bias-no-normalize";
  argv[6] = (char*) "--no-final-arc-feats";
  argv[7] = (char*) "--estimate-learning-rate";
  argv[8] = (char*) "--max-iters=50";
  argv[9] = (char*) "--report-validation-stats=10";
  
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
  BOOST_REQUIRE_EQUAL(d, 281);
  
  // Set the weights to random values.
  Parameters theta = objective->getDefaultParameters(d);
  shared_array<double> samples = Utility::generateGaussianSamples(d, 0, 1);
  theta.w.setWeights(samples.get(), d);
  
  const double beta = 1e-8;
  shared_ptr<Regularizer> l2(new RegularizerL2(beta));
  BOOST_REQUIRE_EQUAL(l2->getBeta(), beta);
  StochasticGradientOptimizer opt(objective, l2);
  ret = opt.processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret); 
  
  double fval;
  Optimizer::status status = opt.train(theta, fval, 1e-5);
  BOOST_REQUIRE(status == Optimizer::CONVERGED);
  
  SparseRealVec gradFv(d);
  objective->valueAndGradient(theta, fval, gradFv);
  l2->addRegularization(theta, fval, gradFv);
  BOOST_CHECK_CLOSE(1.06272359235157, fval, 1e-8);
}
