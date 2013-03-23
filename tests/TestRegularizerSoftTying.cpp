#define BOOST_TEST_DYN_LINK

#include "Alphabet.h"
#include "Label.h"
#include "Parameters.h"
#include "RegularizerSoftTying.h"
#include "Ublas.h"
#include "Utility.h"
#include <boost/lexical_cast.hpp> 
#include <boost/shared_array.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost;
using namespace std;

BOOST_AUTO_TEST_CASE(testRegularizerSoftTying)
{
  const int argc = 5;
  char* argv[argc];
  argv[0] = (char*) "latent_struct";
  argv[1] = (char*) "--w-beta=0.1";
  argv[2] = (char*) "--shared-w-beta=1";
  argv[3] = (char*) "--u-beta=0.5";
  argv[4] = (char*) "--shared-u-beta=2";
  
  const int numLabels = 3;
  set<Label> labels;
  for (Label y = 0; y < numLabels; ++y)
    labels.insert(y);
  
  Alphabet alphabet;
  const int numFeatures = 5;
  set<Label>::const_iterator it;
  for (it = labels.begin(); it != labels.end(); ++it) {
    for (int featnum = 0; featnum < numFeatures; ++featnum) {
      const string f = "feature" + lexical_cast<string>(featnum);
      alphabet.lookup(f, *it, true);
    }
  }  
  BOOST_CHECK_EQUAL(alphabet.size(), numFeatures * labels.size());
  BOOST_CHECK_EQUAL(alphabet.numFeaturesPerClass(), numFeatures);
  
  Parameters theta(alphabet.size(), alphabet.size());
  RegularizerSoftTying regularizer;
  regularizer.processOptions(argc, argv);
  regularizer.setupParameters(theta, alphabet, labels);
  alphabet.lock();
  // The call to setupParameters should add a dummy label to alphabet ...
  BOOST_CHECK_EQUAL(alphabet.size(), numFeatures * (labels.size() + 1));
  // ... but the number of explicit labels should not change.
  BOOST_CHECK_EQUAL(labels.size(), numLabels);
  BOOST_CHECK_EQUAL(theta.getTotalDim(), labels.size() * numFeatures * 2);
  const int dim = numFeatures * labels.size();
  BOOST_CHECK_EQUAL(theta.w.getDim(), dim);
  BOOST_CHECK_EQUAL(theta.u.getDim(), dim);
  
  const int classDim = theta.w.getDim();
  const int sharedDim = theta.shared_w.getDim();
  shared_array<double> randomWeights = Utility::generateGaussianSamples(
      classDim + sharedDim, 0, 2, 0);
      
  // Normally, we'd just set the weights as follows. But for the sake of
  // compatibility with an older version of this test (when the Parameters
  // layout was [w shared_w u shared_u] vs. [w u shared_w shared_w] now), we go
  // through the trouble of mapping the same set of initial weights to the new
  // layout.
  // theta.setWeights(randomWeights.get(), theta.getGradientDim());
  
  theta.w.setWeights(randomWeights.get(), classDim);
  theta.shared_w.setWeights(randomWeights.get() + classDim, sharedDim);
  randomWeights = Utility::generateGaussianSamples(classDim + sharedDim, 0, 1,
      1);
  theta.u.setWeights(randomWeights.get(), classDim);
  theta.shared_u.setWeights(randomWeights.get() + classDim, sharedDim);
  
  double fval = 0;
  RealVec grad(theta.getGradientDim());
  grad.clear();
  regularizer.addRegularization(theta, fval, grad);
  
  BOOST_CHECK_CLOSE(fval, 34.789210713823, 1e-8);
  const double checkedGrad[40] = {
      -0.212627,-0.236395,-0.0475705,-0.207021,-0.467987,
      -0.194789,-0.527839,-0.0717731,-0.0785243,-0.0233812,
      -0.0768941,-0.311786,-0.119205,0.233733,-0.0992638,
      -2.10001,1.78653,-0.922543,-0.201966,0.495507,
      -0.61391,0.331119,1.05686,1.16074,0.526089,
      -0.193771,1.20854,-0.514146,1.39526,-0.186833,
      0.054569,2.63049,-0.122656,-1.64313,2.11468,
      5.36409,-7.05839,3.209,-6.11022,-1.76948
  };
  for (size_t i = 0; i < 40; ++i)
    BOOST_CHECK_CLOSE(grad[i], checkedGrad[i], 1e-3);
}
