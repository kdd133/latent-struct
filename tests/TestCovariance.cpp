#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "Model.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "WordAlignmentFeatureGen.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
using boost::numeric::ublas::matrix;
using namespace boost;

/* This test is derived from testStringEditHypergraphVarSemi, but here we are
 * specifically interested in testing ublas_util::computeLowerCovarianceMatrix.
 * The key difference with this test is that we construct an input string pair
 * such that at least one pair of features will never co-occur. In this case,
 * Substitution will never appear in an alignment with Insert or Delete, for
 * example. This exposed a bug in the computeLowerCovarianceMatrix function,
 * which iterated over only the non-zero entries in the co-occurrence matrix,
 * and would yield an erroneous covariance matrix in cases where entry (i,j) of
 * the co-occurrence matrix was zero, but the (i,j) entry in phiBar*phiBar' was
 * non-zero. Recall that Cov = Cooc - phiBar*phiBar'.
 */
BOOST_AUTO_TEST_CASE(testCovariance)
{
  const int argc = 9;
  char* argv[argc];
  size_t i = 0;
  argv[i++] = (char*) "latent_struct";
  argv[i++] = (char*) "--order=0";
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
  Model* model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
  ret = model->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  
  std::vector<Model*> models;
  models.push_back(model);
  
  StringPair* pair = new StringPair("t", "u");
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
  BOOST_REQUIRE_EQUAL(d, 4);
  
  WeightVector W(d);  
  const int iDel = alphabet->lookup("S:Del1", 0, false);
  BOOST_REQUIRE(iDel >= 0);
  W.add(iDel, -1);  
  const int iIns = alphabet->lookup("S:Ins1", 0, false);
  BOOST_REQUIRE(iIns >= 0);
  W.add(iIns, -2);
  const int iSub = alphabet->lookup("S:Sub11", 0, false);
  BOOST_REQUIRE(iSub >= 0);
  W.add(iSub, -3);
  const int iBias = alphabet->lookup("Bias", 0, false);
  BOOST_REQUIRE(iBias >= 0);
  
  SparseLogVec fv(d);
  AccumLogMat fm(d, d);
  LogWeight totalMass = model->expectedFeatureCooccurrences(W, &fm, &fv, *pair,
      label, false);
     
  SparseRealVec phiBar(d);
  ublas_util::exponentiate(fv, phiBar);
  
  AccumRealMat cov(d, d);
  ublas_util::computeLowerCovarianceMatrix(fm, phiBar, cov);

  // Check that the total mass is correct.
  BOOST_CHECK_CLOSE((double)totalMass, -1.90138771133189, 1e-8);
  
  // Check that the (unnormalized) expected value of each feature is correct.  
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iIns]), -2.306852, 1e-4);
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iDel]), -2.306852, 1e-4);
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iSub]), -3.000000, 1e-4);
  BOOST_CHECK_CLOSE((double)((LogWeight)fv[iBias]), -1.901387, 1e-4);
  
  // Check that the entries in the (log) cooccurrence matrix are correct.
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iBias,iBias)), -1.90138771133, 1e-3);
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iDel,iBias)), -2.30685281944, 1e-3);
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iIns,iBias)), -2.30685281944, 1e-3);
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iSub,iBias)), -3, 1e-3);
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iDel,iDel)), -2.30685281944, 1e-3);
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iIns,iDel)), -2.30685281944, 1e-3);
  BOOST_CHECK(isinf((double)((LogWeight)fm(iDel,iSub))));
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iIns,iIns)), -2.30685281944, 1e-3);
  BOOST_CHECK(isinf((double)((LogWeight)fm(iIns,iSub))));
  BOOST_CHECK_CLOSE((double)((LogWeight)fm(iSub,iSub)), -3, 1e-3);
  
  // Check that the entries in the covariance matrix are correct.
  BOOST_CHECK_CLOSE((double)(cov(iBias,iBias)), 0.1270522, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iDel,iBias)), 0.0847019, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iIns,iBias)), 0.0847019, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iSub,iBias)), 0.0423508, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iDel,iDel)), 0.0896594, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iIns,iDel)), 0.0896594, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iSub,iDel)), -0.0049575, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iIns,iIns)), 0.0896594, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iSub,iIns)), -0.0049575, 1e-3);
  BOOST_CHECK_CLOSE((double)(cov(iSub,iSub)), 0.0473083, 1e-3);
}
