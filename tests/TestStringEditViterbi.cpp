#define BOOST_TEST_DYN_LINK

#include "AlignmentTransducer.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "Dataset.h"
#include "LogLinearMulti.h"
#include "Model.h"
#include "StringEditModel.h"
#include "Ublas.h"
#include "WordAlignmentFeatureGen.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>
using namespace boost;

/*
 * Build a first-order edit distance transducer, but only fire zero-order
 * features. Run the Viterbi algorithm to compute the weight of the max-scoring
 * path through the graph, as well as the feature counts from this path.
 * The reason we build the graph in this manner is to duplicate the behaviour
 * of an existing implementation whose dynamic programming routines have
 * been verified to be correct. Note that we can change --order=1 to --order=0
 * below and still get the same result.
 */
BOOST_AUTO_TEST_CASE(testStringEditViterbi)
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
  Model* model = new StringEditModel<AlignmentTransducer<StdFeatArc> >(fgenLat,
      fgenObs);
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
  
  // Check that the Viterbi score is correct.
  double viterbiScore = model->viterbiScore(W, *pair, label);
  BOOST_CHECK_CLOSE((double)viterbiScore, -300, 1e-8);
  
  SparseRealVec fv(d);
  double maxScore = model->maxFeatures(W, fv, *pair, label, true);
  BOOST_CHECK_CLOSE((double)maxScore, (double)viterbiScore, 1e-8);
  
  const int iMat = alphabet->lookup("0_S:Mat11");
  BOOST_REQUIRE(iMat >= 0);
  const int iBias = alphabet->lookup("0_Bias");
  BOOST_REQUIRE(iBias >= 0);
  
  // Check that values in the max-scoring feature vector are correct.  
  BOOST_CHECK_CLOSE((double)fv[iIns], 3, 1e-4);
  BOOST_CHECK_SMALL((double)fv[iDel], 1e-4);
  BOOST_CHECK_SMALL((double)fv[iSub], 1e-4);
  BOOST_CHECK_CLOSE((double)fv[iMat], 6, 1e-4);
  BOOST_CHECK_CLOSE((double)fv[iBias], 1, 1e-4);
  
  // Check that the max-scoring alignment is correct.
  stringstream alignmentStr;
  model->printAlignment(alignmentStr, W, *pair, label);
  const string alignment = alignmentStr.str();
  string correctAlignment =
      "Mat11 Ins1 Mat11 Ins1 Mat11 Mat11 Mat11 Mat11 Ins1 \n";
  correctAlignment += "|s| |t| |r|e|s|s| \n|s|u|t|o|r|e|s|s|u\n";
  BOOST_CHECK(alignment == correctAlignment);
}
