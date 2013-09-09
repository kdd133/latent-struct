/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Alphabet.h"
#include "BergsmaKondrakPhrasePairs.h"
#include "BergsmaKondrakWordFeatureGen.h"
#include "BiasFeatureGen.h"
#include "Label.h"
#include "StringPairPhrases.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

BergsmaKondrakPhrasePairs::BergsmaKondrakPhrasePairs(
    boost::shared_ptr<Alphabet> alphabet, bool normalize) :
    ObservedFeatureGen(alphabet), _normalize(normalize), _addMismatches(true),
    _addBias(true), _addNed(false) {
}

int BergsmaKondrakPhrasePairs::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  bool noBias = false;
  bool noNormalize = false;
  bool noMismatches = false;
  options.add_options()
    ("bk-ned", opt::bool_switch(&_addNed),
        "include the normalized edit distance feature")
    ("bk-no-bias", opt::bool_switch(&noBias), "do not add a bias feature")
    ("bk-no-mismatches", opt::bool_switch(&noMismatches),
        "do not include mismatch features")
    ("bk-no-normalize", opt::bool_switch(&noNormalize),
        "do not normalize by the length of the longer word")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help")) {
    cout << options << endl;
    return 0;
  }
  
  if (noBias)
    _addBias = false;
  if (noMismatches)
    _addMismatches = false;
  if (noNormalize)
    _normalize = false;

  return 0;
}

SparseRealVec* BergsmaKondrakPhrasePairs::getFeatures(const Pattern& x,
    const Label y) {
  const StringPairPhrases& pair = (const StringPairPhrases&)x;
  
  const vector<string>& pp = pair.getPhrasePairs();
  assert(pp.size() > 0);

  unordered_map<int, int> phrasePairCounts;

  BOOST_FOREACH(const string& phrasePair, pp) {
    // Note: At test time, the alphabet will presumably be locked, so even though
    // addIfAbsent=true, features not seen during training won't be added/used.
    int fId = _alphabet->lookup(phrasePair, y, true);
    if (fId >= 0)
      phrasePairCounts[fId]++;
  }
  
  if (_addMismatches) {
    BergsmaKondrakWordFeatureGen::getMismatches(pair.getSource(),
        pair.getTarget(), phrasePairCounts, y, *_alphabet, true);
  }
  
  // Add a bias feature if this option is enabled.
  // TODO: In the future, it would be cleaner to be able to activate multiple
  // feature generators via command line options. Here, we essentially have
  // a BiasFeatureGen inside of a BergsmaKondrakPhrasePairs.
  if (_addBias) {
    const int fId = _alphabet->lookup(BiasFeatureGen::kPrefix, y, true);
    if (fId >= 0)
      phrasePairCounts[fId] = 1;
  }
  
  const size_t d = _alphabet->size();
  SparseRealVec* fv = new SparseRealVec(d);
  
  unordered_map<int, int>::const_iterator it;
  for (it = phrasePairCounts.begin(); it != phrasePairCounts.end(); ++it) {
    const size_t index = it->first;
    if (index >= d) {
      // Assume we're in feature gathering mode, in which case the fv we return
      // will be ignored.
      return fv;
    }
    (*fv)(index) = it->second;
  }
  
  if (_addNed) {
    // Include a normalized edit distance feature, which is defined as the edit
    // distance divided by the length of the longer word.
    double ned = pair.getEditDistance();
    // If we're normalizing, this feature will end up being normalized twice,
    // so we skip normalization here in that case. 
    if (!_normalize) {
      const double normalization = pair.getSize();
      assert(normalization > 0);
      ned /= normalization;
    }
    const int fId = _alphabet->lookup(BergsmaKondrakWordFeatureGen::NED_FEATURE,
        y, true);
    if (fId >= 0)
      (*fv)[fId] = ned;
  }
  
  if (_normalize) {
    const double normalization = pair.getSize();
    assert(normalization > 0);
    (*fv) /= normalization;
  }

  return fv;
}
