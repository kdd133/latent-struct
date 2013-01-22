#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BmrmOptimizer.h"
#include "Dataset.h"
#include "EmptyAlignmentFeatureGen.h"
#include "KlementievRothWordFeatureGen.h"
#include "MaxMarginBinaryObs.h"
#include "Parameters.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "Utility.h"
#include "WordPairReader.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
using namespace boost;


BOOST_AUTO_TEST_CASE(testMaxMarginBinaryObs)
{
  const int argc = 2;
  char* argv[argc];
  argv[0] = (char*) "latent_struct";
  argv[1] = (char*) "--quiet";
  
  shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
  shared_ptr<KlementievRothWordFeatureGen> fgenObs(
      new KlementievRothWordFeatureGen(alphabet));
  int ret = fgenObs->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  shared_ptr<EmptyAlignmentFeatureGen> fgenLat(new EmptyAlignmentFeatureGen(
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
  
  MaxMarginBinaryObs objective(trainData, models);
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective.gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int d = alphabet->size();
  BOOST_REQUIRE_EQUAL(d, 686);
  
  Parameters W = objective.getDefaultParameters(d);
  
  // set the feature weight for bias class y=1 to one
  int index = alphabet->lookup("1_Bias", false);
  BOOST_REQUIRE(index >= 0);
  W.add(index, 1.0);
  BOOST_REQUIRE_EQUAL(W.getWeight(index), 1.0);
  
  RealVec gradFv(d);
  double fval;
  objective.valueAndGradient(W, fval, gradFv);
  BOOST_CHECK_CLOSE(0.96922619047, fval, 1e-8);
  BOOST_CHECK_CLOSE(-0.0125, gradFv[0], 1e-6);
  BOOST_CHECK_CLOSE(-0.0166666666, gradFv[1], 1e-6);
  BOOST_CHECK_CLOSE(-0.0011904761, gradFv[280],1e-4);
  BOOST_CHECK_CLOSE(0.0125, gradFv[362], 1e-4);
  
  BmrmOptimizer opt(objective);
  ret = opt.processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret);
  opt.setBeta(0.1);
  BOOST_REQUIRE_EQUAL(opt.getBeta(), 0.1);
  
  double fvalOpt = 0.0;
  Optimizer::status status = opt.train(W, fvalOpt, 1e-5);
  BOOST_REQUIRE(status == Optimizer::CONVERGED);
  
  objective.valueAndGradient(W, fval, gradFv);
  Utility::addRegularizationL2(W, opt.getBeta(), fval, gradFv);
  BOOST_CHECK_CLOSE(0.503785233955, fval, 1e-8);
}
