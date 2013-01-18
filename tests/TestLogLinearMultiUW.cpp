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
  argv[9] = (char*) "--lbfgs-no-regularization";
  
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
  LogLinearMultiUW objective(trainData, models); // objective now owns models
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective.gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int numFeats = alphabet->size();
  BOOST_REQUIRE_EQUAL(numFeats, 8);
  
  Parameters theta(numFeats, numFeats);
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
  LogLinearMulti objectiveW(trainData, modelsW);
  Parameters thetaW(numFeats);
  BOOST_REQUIRE(!thetaW.hasU());
  thetaW.setWeights(samples.get(), numFeats);
  
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
  
  const double tol = 1e-5;
  double fvalOpt = 0.0;
  {
    LbfgsOptimizer opt(objective);
    ret = opt.processOptions(argc, argv);
    BOOST_REQUIRE_EQUAL(0, ret);
    Optimizer::status status = opt.train(theta, fvalOpt, tol);
    BOOST_REQUIRE(status == Optimizer::CONVERGED);
  }  
  objective.valueAndGradient(theta, fval, gradFv);
//  Utility::addRegularizationL2(theta, opt.getBeta(), fval, gradFv);
  BOOST_CHECK_CLOSE(fval, fvalOpt, 1e-8);
  BOOST_CHECK_CLOSE(fval, 0.67628998419572, 1e-8);

  double fvalOptW = 0.0;
  {
    LbfgsOptimizer opt(objectiveW);
    ret = opt.processOptions(argc, argv);
    BOOST_REQUIRE_EQUAL(0, ret);
    Optimizer::status status = opt.train(thetaW, fvalOptW, tol);
    BOOST_REQUIRE(status == Optimizer::CONVERGED);
  }
  
  BOOST_CHECK_CLOSE(fvalOpt, fvalOptW, 1e-8); // pass if we decrease tol?

  // Shouldn't w be approximately equal to u at the optimizer?
  BOOST_REQUIRE_EQUAL(theta.w.getDim(), theta.u.getDim());
  for (int i = 0; i < theta.w.getDim(); ++i)
    BOOST_CHECK_CLOSE(theta.w.getWeight(i), theta.u.getWeight(i), 1e-8);

  using namespace std;
  cout << "w: " << theta.w << endl << "u: " << theta.u << endl;

  theta.u.setWeights(theta.w.getWeights(), theta.w.getDim());
  objective.valueAndGradient(theta, fval, gradFv);
  BOOST_CHECK_CLOSE(fval, fvalOpt, 1e-8);
//  Utility::addRegularizationL2(theta, opt.getBeta(), fval, gradFv);
  BOOST_CHECK_CLOSE(fval, 0.67628998419572, 1e-8);
  
  cout << endl << "w: " << theta.w << endl << "u: " << theta.u << endl;
    

}
