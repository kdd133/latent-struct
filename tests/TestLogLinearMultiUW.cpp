#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "Example.h"
#include "Label.h"
#include "LabelScoreTable.h"
#include "LbfgsOptimizer.h"
#include "LogLinearMulti.h"
#include "LogLinearMultiUW.h"
#include "Parameters.h"
#include "RegularizerNone.h"
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
  shared_ptr<TrainingObjective> objective(new LogLinearMultiUW(trainData,
      models)); // Note: objective now owns models
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective->gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int numFeats = alphabet->size();
  BOOST_REQUIRE_EQUAL(numFeats, 8);
  
  Parameters theta = objective->getDefaultParameters(numFeats);
  const int d = theta.getTotalDim();
  
  // Set the weights to random values, but set the w and u weights to be equal.
  shared_array<double> samples = Utility::generateGaussianSamples(numFeats, 0, 1);
  theta.w.setWeights(samples.get(), numFeats);
  theta.u.setWeights(samples.get(), numFeats);
  
  // Create a LogLinearMulti objective and initialize parameters thetaW such
  // that thetaW.w == theta.w
  std::vector<Model*> modelsW;
  {
    // We need to create separate (although identical) models, since the
    // LogLinearMultiUW objective above owns the vector named "models".
    Model* model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
    ret = model->processOptions(argc, argv);
    BOOST_REQUIRE_EQUAL(ret, 0);
    modelsW.push_back(model);
  }
  shared_ptr<TrainingObjective> objectiveW(new LogLinearMulti(trainData,
      modelsW));
  Parameters thetaW(numFeats);
  BOOST_REQUIRE(!thetaW.hasU());
  thetaW.setWeights(samples.get(), numFeats);
  
  // Get the function value and gradient for LogLinearMulti.
  RealVec gradFvW(numFeats);
  double fvalW;
  objectiveW->valueAndGradient(thetaW, fvalW, gradFvW);
  
  // Get the function value and gradient for LogLinearMultiUW. 
  RealVec gradFv(d);
  double fval;
  objective->valueAndGradient(theta, fval, gradFv);
  BOOST_CHECK_CLOSE(fval, 1.7556439545048077, 1e-8);
  const double checkedGrad[8] = { 0.46434287641057681, 2.0387147255979778,
      1.9056367355983204, 0.31683516398689121, -0.46434287641057664,
      -0.6515068509204448, -0.51842886092062412, -1.7040430386649665
  };
  for (int i = 0; i < theta.w.getDim(); ++i)
    BOOST_CHECK_CLOSE(gradFv[i], checkedGrad[i], 1e-8);
  
  // Since we set w == u above, the function values and the w portion of the
  // gradients should be equal.
  BOOST_CHECK_CLOSE(fvalW, fval, 1e-8);
  for (int i = 0; i < theta.w.getDim(); ++i)
    BOOST_CHECK_CLOSE(gradFvW[i], gradFv[i], 1e-8);

  shared_ptr<Regularizer> reg(new RegularizerNone());

  // Find the optimal (w,u) parameters for LogLinearMultiUW.
  const double tol = 1e-6;
  double fvalOpt = 0.0;
  {
    LbfgsOptimizer opt(objective, reg);
    ret = opt.processOptions(argc, argv);
    BOOST_REQUIRE_EQUAL(0, ret);
    Optimizer::status status = opt.train(theta, fvalOpt, tol);
    BOOST_CHECK_EQUAL(status, Optimizer::CONVERGED);
    BOOST_REQUIRE_EQUAL(theta.w.getDim(), theta.u.getDim());
  }  
  objective->valueAndGradient(theta, fval, gradFv);
  BOOST_CHECK_CLOSE(fval, fvalOpt, 1e-8);
  BOOST_CHECK_CLOSE(fval, 0.433288532370921, 1e-8);

  // Find the optimal w parameters for LogLinearMulti.
  double fvalOptW = 0.0;
  {
    LbfgsOptimizer opt(objectiveW, reg);
    ret = opt.processOptions(argc, argv);
    BOOST_REQUIRE_EQUAL(0, ret);
    Optimizer::status status = opt.train(thetaW, fvalOptW, tol);
    BOOST_CHECK_EQUAL(status, Optimizer::CONVERGED);
  }
  
  // With regularization disabled, fvalOpt and fvalOptW should be equal.
  // Note: This test will pass at 1e-8 if we lower tol to 1e-8, but training is
  // considerably slower.
  BOOST_CHECK_CLOSE(fvalOpt, fvalOptW, 1e-5);
  
  // Shouldn't w be approximately equal to u at the optimizer?
  //  for (int i = 0; i < theta.w.getDim(); ++i)
  //    BOOST_CHECK_CLOSE(theta.w.getWeight(i), theta.u.getWeight(i), 1e-5);
  // Update: In practice, this relationship does not seem to hold. We thus
  // conclude that at the optimum, u isn't that important. That is, we can move
  // around in u and get roughly the same objective value (the space is
  // relatively flat along the u dimension at the optimum). We verify this by
  // setting u=w in the test below and making sure the objective value does not
  // decrease, and finally we set u=0 and check that the value does increase.

  // Count the number of prediction errors that each model makes on the training
  // data. We expect them to be the same.
  const int t = trainData.numExamples();
  const int k = trainData.getLabelSet().size();
  LabelScoreTable yHat(t, k);
  objective->predict(theta, trainData, yHat);
  LabelScoreTable yHatW(t, k);
  objectiveW->predict(thetaW, trainData, yHatW);  
  int errors = 0, errorsW = 0;  
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
    if (yHatW.getScore(id, 0) > yHatW.getScore(id, 1)) {
      if (yi == 1)
        errorsW++;
    }
    else {
      if (yi == 0)
        errorsW++;
    }
  }  
  BOOST_CHECK_EQUAL(errors, 4);
  BOOST_CHECK_EQUAL(errors, errorsW);

  // If we now set u=w (using the optimal w found above), we should not
  // observe a change in the objective value. If we did, it would mean that
  // our optimization algorithm is not finding the best (w,u) parameters.
  double fval_setUtoW = 0.0;
  theta.u.setWeights(theta.w.getWeights(), theta.w.getDim());
  objective->valueAndGradient(theta, fval_setUtoW, gradFv);
  BOOST_CHECK_CLOSE(fvalOpt, fval_setUtoW, 1e-5);
  
  // Just as a sanity check, let's set u=0 and check that the function value
  // increases. If this weren't the case, u would appear to be meaningless!
  double fval_zeroU;
  theta.u.zero();
  objective->valueAndGradient(theta, fval_zeroU, gradFv);
  BOOST_CHECK(fval_zeroU > 2 * fvalOpt); // let's look for at least a doubling
}
