#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LbfgsOptimizer.h"
#include "LogLinearMulti.h"
#include "LogLinearMultiUW.h"
#include "Parameters.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "Utility.h"
#include "WordAlignmentFeatureGen.h"
#include "WordPairReader.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost;

BOOST_AUTO_TEST_CASE(testLogLinearMultiUW)
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
  Model* model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
  ret = model->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  
  std::vector<Model*> models;
  models.push_back(model);  
  
  Dataset trainData;
  WordPairReader reader;
  bool badFile = Utility::loadDataset(reader, "he_tiny", trainData);
  BOOST_REQUIRE(!badFile);
  
  LogLinearMultiUW objective(trainData, models);
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective.gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int numFeats = alphabet->size();
  BOOST_REQUIRE_EQUAL(numFeats, 8);
  
  Parameters theta(numFeats, numFeats);
  const int d = theta.getTotalDim();
  
  // Set the weights to some random values.
//  shared_array<double> samples = Utility::generateGaussianSamples(d, 0, 1);
//  theta.setWeights(samples.get(), d);
//  samples.reset();
  
  // Set the weights to random values, but set the w and u weights to be equal.
//  const int n = theta.w.getDim();
//  shared_array<double> samples = Utility::generateGaussianSamples(n, 0, 1);
//  theta.w.setWeights(samples.get(), n);
//  theta.u.setWeights(samples.get(), n);
//  samples.reset();

  // set the feature weight for bias class y=0 to one
  int index = alphabet->lookup("0_Bias", false);
  BOOST_REQUIRE(index >= 0);
  theta.w.add(index, 1.0);
  theta.u.add(index, 1.0);
  
  // Create a LogLinearMulti objective and initialize parameters thetaW such
  // that thetaW.w == theta.w
  LogLinearMulti objectiveW(trainData, models); // inherit existing alphabet
  Parameters thetaW(numFeats);
  BOOST_REQUIRE(!thetaW.hasU());
  thetaW.add(index, 1.0);
  
  // Get the function value and gradient for LogLinearMulti.
  RealVec gradFvW(numFeats);
  double fvalW;
  objectiveW.valueAndGradient(thetaW, fvalW, gradFvW);
  
  // Get the function value and gradient for LogLinearMultiUW. 
  RealVec gradFv(d);
  double fval;
  objective.valueAndGradient(theta, fval, gradFv);
  
  // Since we set w == u above, the function values and the w portion of the
  // gradients should be equal.
  BOOST_CHECK_CLOSE(fvalW, fval, 1e-8);
  for (int i = 0; i < theta.w.getDim(); ++i)
    BOOST_CHECK_CLOSE(gradFvW[i], gradFv[i], 1e-8);
  
  LbfgsOptimizer opt(objective);
  ret = opt.processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret);
  opt.setBeta(0.1);
  BOOST_REQUIRE_EQUAL(opt.getBeta(), 0.1);
  
  double fvalOpt = 0.0;
  Optimizer::status status = opt.train(theta, fvalOpt, 1e-5);
  BOOST_REQUIRE(status == Optimizer::CONVERGED);
  
  objective.valueAndGradient(theta, fval, gradFv);
  Utility::addRegularizationL2(theta, opt.getBeta(), fval, gradFv);
  BOOST_CHECK_CLOSE(0.67414588974, fval, 1e-8);
}
