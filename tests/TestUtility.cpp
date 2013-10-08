#define BOOST_TEST_DYN_LINK

#include "Utility.h"
#include <string>
#include <vector>

#include <boost/test/unit_test.hpp>

using namespace boost;
using namespace std;

BOOST_AUTO_TEST_CASE(testUtility)
{
  vector<string> source, target, sourceAligned, targetAligned;
  
  target.push_back("k");
  target.push_back("i");
  target.push_back("t");
  target.push_back("t");
  target.push_back("e");
  target.push_back("n");
  
  source.push_back("s");
  source.push_back("i");
  source.push_back("t");
  source.push_back("t");
  source.push_back("i");
  source.push_back("n");
  source.push_back("g");
  
  int cost = Utility::levenshtein(source, target, sourceAligned, targetAligned,
    3);
  BOOST_CHECK_EQUAL(cost, 5);
  BOOST_CHECK_EQUAL(sourceAligned.size(), targetAligned.size());

  const string desiredSource = "s-itti-ng";
  const string desiredTarget = "-kitt-en-";

  // Conver each vector of strings representing an alignment into one string. 
  string alignedSource, alignedTarget;
  for (size_t i = 0; i < sourceAligned.size(); ++i) {
    alignedSource += sourceAligned[i];
    alignedTarget += targetAligned[i];
  }
  for (size_t i = 0; i < sourceAligned.size(); ++i)
    BOOST_CHECK(alignedSource[i] == desiredSource[i]);
  for (size_t j = 0; j < targetAligned.size(); ++j)
    BOOST_CHECK(alignedTarget[j] == desiredTarget[j]);
}
