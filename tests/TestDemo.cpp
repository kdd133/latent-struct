#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestDemo

#include "AlignmentTransducer.h"
#include "Alphabet.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "BmrmOptimizer.h"
#include "Dataset.h"
#include "EditOperation.h"
#include "EmptyAlignmentFeatureGen.h"
#include "EmptyObservedFeatureGen.h"
#include "FeatureVector.h"
#include "FeatureVector.h"
#include "KlementievRothWordFeatureGen.h"
#include "LbfgsOptimizer.h"
#include "LogFeatArc.h"
#include "LogLinearBinary.h"
#include "LogLinearBinaryObs.h"
#include "LogLinearMulti.h"
#include "LogWeight.h"
#include "OpDelete.h"
#include "OpInsert.h"
#include "OpMatch.h"
#include "OpReplace.h"
#include "OpSubstitute.h"
#include "Optimizer.h"
#include "RealWeight.h"
#include "StateType.h"
#include "StdFeatArc.h"
#include "WordAlignmentFeatureGen.h"
#include "StringEditModel.h"
#include "StringPair.h"
#include "Utility.h"
#include "WeightVector.h"
#include "WordAlignmentFeatureGen.h"
#include "WordPairReader.h"
#include <boost/foreach.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <fst/vector-fst.h>
#include <list>
#include <stdexcept>
#include <string>


BOOST_AUTO_TEST_CASE(testStringEdit)
{
  const int argc = 8;
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
  
  shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
  shared_ptr<BiasFeatureGen> fgenObs(new BiasFeatureGen(alphabet));
  int ret = fgenObs->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  shared_ptr<WordAlignmentFeatureGen> fgenLat(new WordAlignmentFeatureGen(
      alphabet));
  ret = fgenLat->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  Model* model = new StringEditModel<LogFeatArc>(fgenLat, fgenObs);
  ret = model->processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(ret, 0);
  
  vector<Model*> models;
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
  BOOST_REQUIRE_EQUAL(d, 4);
  
  WeightVector W(d);
  
  int fid = alphabet->lookup("0_S:Del1");
  BOOST_REQUIRE(fid >= 0);
  W.add(fid, -100);  
  fid = alphabet->lookup("0_S:Ins1");
  BOOST_REQUIRE(fid >= 0);
  W.add(fid, -100);
  fid = alphabet->lookup("0_S:Rep11");
  BOOST_REQUIRE(fid >= 0);
  W.add(fid, -100);
  
  LogWeight totalMass = model->totalMass(W, *pair, label);
  BOOST_CHECK_CLOSE(totalMass.value(), -295.5691832011567, 1e-8);
  
//  shared_ptr<Alphabet> alphabet(new Alphabet());
//  shared_ptr<EmptyObservedFeatureGen> empty(new EmptyObservedFeatureGen(
//      alphabet));
//  WeightVector wNull;
//  const int label = 1;
//  
//  list<StateType> states;
//  states.push_back(StateType(0, "start"));
//  states.push_back(StateType(1, "ins"));
//  states.push_back(StateType(2, "del"));
//  states.push_back(StateType(3, "idn"));
//  states.push_back(StateType(4, "sub"));
//  
//  OpInsert opInsert(1, 1);
//  OpDelete opDelete(2, 2);
//  OpMatch opMatch(3, 3);
//  OpSubstitute opSubstitute(4, 4);
//    
//  list<const EditOperation*> ops;
//  ops.push_back(&opDelete);
//  ops.push_back(&opInsert);
//  ops.push_back(&opSubstitute);
//  ops.push_back(&opMatch);
//  
//  StringPair pair("stress", "sutoressu");
//  
//  shared_ptr<StringAlignmentFeatureGen> fgen(new StringAlignmentFeatureGen(
//      alphabet, 1, true, true, true, false));
//  shared_ptr<BiasFeatureGen> bias(new BiasFeatureGen(alphabet));
//  AlignmentTransducer<StdFeatArc> gatherTransducer(states, ops, fgen, bias,
//      false);
//  gatherTransducer.build(wNull, pair, label);
//  
//  // Create a weight vector and initialize the values. 
//  const int d = alphabet->size();
//  BOOST_CHECK_EQUAL(58, d);
//  WeightVector wv(d);
//  const int nHighFeats = 3;
//  string highCostFeats[nHighFeats] = {"1_E:Insert", "1_E:Substitute",
//      "1_E:Delete"};
//  BOOST_FOREACH(string f, highCostFeats) {
//    const int i = alphabet->lookup(f, false);
//    wv.add(i, -100);
//  }
//  BOOST_CHECK_CLOSE(nHighFeats*(100*100), wv.squaredL2Norm(), 1e-8);
//  
//  AlignmentTransducer<LogFeatArc> trans(states, ops, fgen, empty, false);
//  trans.build(wv, pair, label);
//  LogWeight Z = trans.logPartition();
//  //trans.toGraphviz("trans.dot");
//  BOOST_CHECK_CLOSE(-300.0, Z, 1e-8);
//  
//  FeatureVector<LogWeight> fv(d, true); 
//  LogWeight ZAlt = trans.logExpectedFeaturesUnnorm(fv);
//  BOOST_CHECK_EQUAL(Z, ZAlt);
//  BOOST_CHECK_CLOSE(-298.9014, fv.getValueAtIndex(alphabet->lookup(
//      "1_E:Insert", false)), 1e-4);
//  
//  // Using the same weight vector, the log partition value should be the same
//  // if we exclude the annoted edit features (since these are given zero weight)
//  shared_ptr<StringAlignmentFeatureGen> fgen2(new StringAlignmentFeatureGen(
//      alphabet, 1, false, true, true, false));
//  AlignmentTransducer<StdFeatArc> gatherTransducer2(states, ops, fgen2, empty,
//    false);
//  gatherTransducer2.build(wNull, pair, label);
//  AlignmentTransducer<LogFeatArc> trans2(states, ops, fgen2, empty, false);
//  trans2.build(wv, pair, label);
//  double Z2 = trans2.logPartition();
//  BOOST_CHECK_CLOSE(Z, Z2, 1e-8);
}

/*
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
  Model* model = new StringEditModel<LogFeatArc>(fgenLat, fgenObs);
  ret = model->processOptions(argc, argv);
  BOOST_CHECK_EQUAL(ret, 0);
  
  vector<Model*> models;
  models.push_back(model);  
  
  Dataset trainData;
  WordPairReader reader;
  bool badFile = Utility::loadDataset(reader, "he_tiny", trainData);
  BOOST_REQUIRE(!badFile);
  
  LogLinearMulti objective(trainData, models);
  size_t maxNumFvs = 0, totalNumFvs = 0;
  objective.gatherFeatures(maxNumFvs, totalNumFvs);
  BOOST_REQUIRE(maxNumFvs > 0 && totalNumFvs > 0);
  
  BOOST_CHECK(!alphabet->isLocked());
  alphabet->lock();
  const int d = alphabet->size();
  BOOST_REQUIRE_EQUAL(d, 8);
  
  WeightVector W(d);
  
  // set the feature weight for bias class y=0 to one
  int index = alphabet->lookup("0_Bias", false);
  BOOST_REQUIRE(index >= 0);
  W.add(index, 1.0);
  BOOST_REQUIRE_EQUAL(W.getWeight(index), 1.0);
  
  FeatureVector<RealWeight> gradFv(d);
  double fval;
  objective.valueAndGradient(W, fval, gradFv);
  BOOST_CHECK_CLOSE(0.81326168751, fval, 1e-8);
  BOOST_CHECK_CLOSE(0.23105857863, gradFv.getValueAtLocation(0).value(), 1e-8);
  BOOST_CHECK_CLOSE(0.71278741450, gradFv.getValueAtLocation(1).value(), 1e-8);
  BOOST_CHECK_CLOSE(0.69725812519, gradFv.getValueAtLocation(2).value(), 1e-8);
  BOOST_CHECK_CLOSE(0.30803476795, gradFv.getValueAtLocation(3).value(), 1e-8);
  BOOST_CHECK_CLOSE(-gradFv.getValueAtLocation(0).value(),
      gradFv.getValueAtLocation(4).value(), 1e-8);
  BOOST_CHECK_CLOSE(-gradFv.getValueAtLocation(1).value(),
      gradFv.getValueAtLocation(5).value(), 1e-8);
  BOOST_CHECK_CLOSE(-gradFv.getValueAtLocation(2).value(),
      gradFv.getValueAtLocation(6).value(), 1e-8);
  BOOST_CHECK_CLOSE(-gradFv.getValueAtLocation(3).value(),
      gradFv.getValueAtLocation(7).value(), 1e-8);
  
  LbfgsOptimizer opt(objective);
  ret = opt.processOptions(argc, argv);
  BOOST_REQUIRE_EQUAL(0, ret);
  opt.setBeta(0.1);
  BOOST_REQUIRE_EQUAL(opt.getBeta(), 0.1);
  
  double fvalOpt = 0.0;
  Optimizer::status status = opt.train(W, fvalOpt, 1e-5);
  BOOST_REQUIRE(status == Optimizer::CONVERGED);
  
  objective.valueAndGradient(W, fval, gradFv);
  Utility::addRegularizationL2(W, opt.getBeta(), fval, gradFv);
  BOOST_CHECK_CLOSE(0.67414588974, fval, 1e-8);
}
*/
