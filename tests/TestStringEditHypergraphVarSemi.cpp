#define BOOST_TEST_DYN_LINK

#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "Model.h"
#include "StringEditModel.h"
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

BOOST_AUTO_TEST_CASE(testStringEditHypergraphVarSemi)
{
  const int argc = 9;
  char* argv[argc];
  size_t i = 0;
  argv[i++] = (char*) "latent_struct";
  argv[i++] = (char*) "--order=1";
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
  
  StringPair* pair = new StringPair("stress", "sutoressu");
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
  BOOST_REQUIRE_EQUAL(d, 5);
  
  // Set the weights of Del, Ins, and Sub to -100; implicitly leave Mat to be 0.
  WeightVector W(d);  
  const int iDel = alphabet->lookup("0_S:Del1");
  BOOST_REQUIRE(iDel >= 0);
  W.add(iDel, -100);  
  const int iIns = alphabet->lookup("0_S:Ins1");
  BOOST_REQUIRE(iIns >= 0);
  W.add(iIns, -100);
  const int iSub = alphabet->lookup("0_S:Sub11");
  BOOST_REQUIRE(iSub >= 0);
  W.add(iSub, -100);
  const int iMat = alphabet->lookup("0_S:Mat11");
  BOOST_REQUIRE(iMat >= 0);
  const int iBias = alphabet->lookup("0_Bias");
  BOOST_REQUIRE(iBias >= 0);
  
  shared_ptr<matrix<LogWeight> > fm;
  shared_ptr<FeatureVector<LogWeight> > fv;
  LogWeight totalMass = model->expectedFeatureCooccurrences(W, fm, fv, *pair,
      label, false);
  BOOST_REQUIRE(fm != 0);
  BOOST_REQUIRE(fv != 0);
  
  // Check that the total mass is correct.
  BOOST_CHECK_CLOSE((double)totalMass, -300, 1e-8);
  
  // Check that the (unnormalized) expected value of each feature is correct.  
  BOOST_CHECK_CLOSE((double)fv->getValueAtIndex(iIns), -298.9014, 1e-4);
  BOOST_CHECK_CLOSE((double)fv->getValueAtIndex(iDel), -497.9206, 1e-4);
  BOOST_CHECK_CLOSE((double)fv->getValueAtIndex(iSub), -398.2082, 1e-4);
  BOOST_CHECK_CLOSE((double)fv->getValueAtIndex(iMat), -298.2082, 1e-4);
  BOOST_CHECK_CLOSE((double)fv->getValueAtIndex(iBias), -300.0000, 1e-4);
  
  // Display the matrix of feature cooccurrences.
//  fm->print(cout);
//  cout << endl;
//  fm->print(cout, *alphabet);
  
  // Check that the entries in the cooccurrence matrix are correct.
  matrix<LogWeight>& featureMatrix = *fm;
  BOOST_CHECK_CLOSE((double)featureMatrix(iBias,iBias), -300, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iBias,iDel), -497.921, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iBias,iIns), -298.901, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iBias,iSub), -398.208, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iBias,iMat), -298.208, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iDel,iBias), -497.921, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iDel,iDel), -497.921, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iDel,iIns), -496.534, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iDel,iSub), -596.068, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iDel,iMat), -496.311, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iIns,iBias), -298.901, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iIns,iDel), -496.534, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iIns,iIns), -297.803, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iIns,iSub), -397.11, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iIns,iMat), -297.11, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iSub,iBias), -398.208, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iSub,iDel), -596.068, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iSub,iIns), -397.11, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iSub,iSub), -398.208, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iSub,iMat), -396.599, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iMat,iBias), -298.208, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iMat,iDel), -496.311, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iMat,iIns), -297.11, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iMat,iSub), -396.599, 1e-3);
  BOOST_CHECK_CLOSE((double)featureMatrix(iMat,iMat), -296.416, 1e-3);
}
