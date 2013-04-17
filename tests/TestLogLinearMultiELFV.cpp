#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "Example.h"
#include "Label.h"
#include "LabelScoreTable.h"
#include "LbfgsOptimizer.h"
#include "LogLinearMultiELFV.h"
#include "Parameters.h"
#include "RegularizerL2.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "Utility.h"
#include "WordAlignmentFeatureGen.h"
#include "WordPairReader.h"
#include <boost/foreach.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost;

BOOST_AUTO_TEST_CASE(testLogLinearMultiELFV)
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
  BOOST_REQUIRE_EQUAL(ret, 0);
  shared_ptr<WordAlignmentFeatureGen> fgenLat(new WordAlignmentFeatureGen(
      alphabet));
  ret = fgenLat->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  
  Dataset trainData;
  WordPairReader reader;
  bool badFile = Utility::loadDataset(reader, "he_tiny", trainData);
  BOOST_REQUIRE(!badFile);
  
  std::vector<Model*> models;
  {
    Model* model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
    ret = model->processOptions(argc, argv);
    BOOST_REQUIRE_EQUAL(ret, 0);
    models.push_back(model);
  }
  shared_ptr<TrainingObjective> objective(new LogLinearMultiELFV(trainData,
      models)); // Note: objective now owns models
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective->gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int numFeats = alphabet->size();
  BOOST_REQUIRE_EQUAL(numFeats, 8);
  
  Parameters theta = objective->getDefaultParameters(numFeats);
  const int d = theta.getDimTotal();
  
  // Set the weights to random values, but set the w and u weights to be equal.
  shared_array<double> samples = Utility::generateGaussianSamples(numFeats, 0, 1);
  theta.w.setWeights(samples.get(), numFeats);
  theta.u.setWeights(samples.get(), numFeats);
  
  // Check the function value and gradient. 
  RealVec gradFv(d);
  double fval;
  objective->valueAndGradient(theta, fval, gradFv);
  BOOST_CHECK_CLOSE(fval, 0.8285285674643136, 1e-7);
  const double checkedGrad[8] = { 0.31396735603783266, 1.3557577692589886,
      1.3077399433587242, 0.21493254964414349, -0.31396735603783271,
      -0.40881101770846434, -0.36079319180820008, -1.1618793011946678
  };
  for (int i = 0; i < theta.w.getDim(); ++i)
    BOOST_CHECK_CLOSE(gradFv[i], checkedGrad[i], 1e-7);

  shared_ptr<Regularizer> reg(new RegularizerL2(0.01));

  // Find the optimal (w,u) parameters for LogLinearMultiELFV.
  const double tol = 1e-4;
  double fvalOpt = 0.0;
  {
    LbfgsOptimizer opt(objective, reg);
    ret = opt.processOptions(argc, argv);
    BOOST_REQUIRE_EQUAL(0, ret);
    Optimizer::status status = opt.train(theta, fvalOpt, tol);
    BOOST_CHECK_EQUAL(status, Optimizer::CONVERGED);
    BOOST_REQUIRE_EQUAL(theta.w.getDim(), theta.u.getDim());
  }  
  BOOST_CHECK_CLOSE(fvalOpt, 0.517689895351953, 1e-6);

  // Count the number of prediction errors that the model makes on the training
  // data. We expect them to be the same.
  const int t = trainData.numExamples();
  const int k = trainData.getLabelSet().size();
  LabelScoreTable yHat(t, k);
  objective->predict(theta, trainData, yHat);
  int errors = 0;
  BOOST_FOREACH(const Example& ex, trainData.getExamples()) {
    const size_t id = ex.x()->getId();
    const Label yi = ex.y();
    if (yHat.getScore(id, 0) > yHat.getScore(id, 1)) {
      if (yi == 1)
        errors++;
    }
    else {
      if (yi == 0)
        errors++;
    }
  }  
  BOOST_CHECK_EQUAL(errors, 5);
  
  // Just as a sanity check, let's set u=0 and check that the function value
  // increases. If this weren't the case, u would appear to be meaningless!
  double fval_zeroU;
  theta.u.zero();
  objective->valueAndGradient(theta, fval_zeroU, gradFv);
  BOOST_CHECK(fval_zeroU > 2 * fvalOpt); // let's look for at least a doubling
}
