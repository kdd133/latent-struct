#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LbfgsOptimizer.h"
#include "LogLinearMulti.h"
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


BOOST_AUTO_TEST_CASE(testLogLinearMulti)
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
  
  shared_ptr<TrainingObjective> objective(new LogLinearMulti(trainData,
      models));
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective->gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int d = alphabet->size();
  BOOST_REQUIRE_EQUAL(d, 8);
  
  Parameters W = objective->getDefaultParameters(d);
  
  // set the feature weight for bias class y=0 to one
  int index = alphabet->lookup("0_Bias", false);
  BOOST_REQUIRE(index >= 0);
  W.add(index, 1.0);
  BOOST_REQUIRE_EQUAL(W.getWeight(index), 1.0);
  
  RealVec gradFv(d);
  double fval;
  objective->valueAndGradient(W, fval, gradFv);
  BOOST_CHECK_CLOSE(0.81326168751, fval, 1e-8);
  BOOST_CHECK_CLOSE(0.23105857863, gradFv[0], 1e-8);
  BOOST_CHECK_CLOSE(0.71278741450, gradFv[1], 1e-8);
  BOOST_CHECK_CLOSE(0.69725812519, gradFv[2], 1e-8);
  BOOST_CHECK_CLOSE(0.30803476795, gradFv[3], 1e-8);
  BOOST_CHECK_CLOSE(-gradFv[0], gradFv[4], 1e-8);
  BOOST_CHECK_CLOSE(-gradFv[1], gradFv[5], 1e-8);
  BOOST_CHECK_CLOSE(-gradFv[2], gradFv[6], 1e-8);
  BOOST_CHECK_CLOSE(-gradFv[3], gradFv[7], 1e-8);
  
  const double beta = 0.1;
  shared_ptr<Regularizer> l2(new RegularizerL2(beta));
  BOOST_REQUIRE_EQUAL(l2->getBeta(), beta);
  
  LbfgsOptimizer opt(objective, l2);
  ret = opt.processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret);
  
  double fvalOpt = 0.0;
  Optimizer::status status = opt.train(W, fvalOpt, 1e-5);
  BOOST_REQUIRE(status == Optimizer::CONVERGED);
  BOOST_CHECK_CLOSE(0.67414588974, fvalOpt, 1e-8);
}
