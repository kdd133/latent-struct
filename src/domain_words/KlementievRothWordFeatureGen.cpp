/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include <boost/program_options.hpp>
#include <tr1/unordered_map>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "FeatureGenConstants.h"
#include "KlementievRothWordFeatureGen.h"
#include "FeatureVector.h"
#include "Label.h"
#include "RealWeight.h"
#include "StringPair.h"

const string KlementievRothWordFeatureGen::CHAR_JOINER = "+";
const string KlementievRothWordFeatureGen::SUB_JOINER = ",";

KlementievRothWordFeatureGen::KlementievRothWordFeatureGen(
    boost::shared_ptr<Alphabet> alphabet, bool normalize) :
    ObservedFeatureGen(alphabet), _substringSize(2), _offsetSize(1),
    _normalize(normalize), _addBias(true), _legacy(false) {
}

int KlementievRothWordFeatureGen::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  bool noBias = false;
  bool noNormalize = false;
  options.add_options()
    ("legacy", opt::bool_switch(&_legacy),
        "perform normalization as in old code")
    ("kr-no-bias", opt::bool_switch(&noBias), "do not add a bias feature")
    ("kr-no-normalize", opt::bool_switch(&noNormalize),
        "do not normalize by the length of the longer word")
    ("offset-size", opt::value<int>(&_offsetSize)->default_value(1),
        "allow substring positions to differ by up to +- this number")
    ("substring-size", opt::value<int>(&_substringSize)->default_value(2),
        "extract substrings up to this length")
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
  
  if (noBias) {
    _addBias = false;
  }  
  if (noNormalize) {
    _normalize = false;
  }
  
  return 0;
}

FeatureVector<RealWeight>* KlementievRothWordFeatureGen::getFeatures(
    const Pattern& x, const Label y) {
  const StringPair& pair = (const StringPair&)x;
  
  // copy the source and target strings into s and t
  // append beginning/end of word markers
  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  assert(s.size() > 0 && t.size() > 0);
  
  const vector<string>* longest; // pointer to the longest string
  const vector<string>* shortest; // pointer to the shortest string

  vector<string> subs_sh; // substrings from the shortest string
  vector<string> subs_lo; // substrings from the longest string
  vector<string>* subs_source; // pointer to the source substrings
  vector<string>* subs_target; // pointer to the target substrings
  tr1::unordered_map<int,RealWeight> sub_pair_counts; // substring pair counts

  // set pointers based on which string is the longest 
  if (s.size() > t.size()) {
    longest = &s;
    shortest = &t;
    subs_source = &subs_lo;
    subs_target = &subs_sh;
  }
  else {
    longest = &t;
    shortest = &s;
    subs_source = &subs_sh;
    subs_target = &subs_lo;
  }

  // extract the k-grams at each position in the shortest string
  for (int i = 0; i < (int) shortest->size(); i++) {
    subs_sh.clear();
    appendSubstrings(shortest, i, _substringSize, shortest->size(), subs_sh);
    subs_lo.clear();

    // extract the k-grams at positions in the longest string that are within
    // _offsetSize of the current position in the shortest string
    for (int j = -_offsetSize; j <= _offsetSize; j++) {
      if (i + j >= 0 && i + j < (int) longest->size())
        appendSubstrings(longest, i + j, _substringSize, longest->size(),
            subs_lo);
    }

    // pair each source substring with each target substring
    for (vector<string>::const_iterator subs_source_it = subs_source->begin();
        subs_source_it != subs_source->end(); subs_source_it++)
      for (vector<string>::const_iterator subs_target_it = subs_target->begin();
          subs_target_it != subs_target->end(); subs_target_it++) {
        stringstream ss;
        ss << y << FeatureGenConstants::PART_SEP << *subs_source_it <<
            SUB_JOINER << *subs_target_it;
        // At test time, the alphabet will presumably be locked, and we don't
        // want to count unseen features; so, we pretend we never saw them.
        const int fId = _alphabet->lookup(ss.str(), true);
        if (fId >= 0)
          sub_pair_counts[fId].plusEquals(RealWeight::kOne);
      }
  }
  
  // Add a bias feature if this option is enabled.
  // TODO: In the future, it would be cleaner to be able to activate multiple
  // feature generators via command line options. Here, we essentially have
  // a BiasFeatureGen inside of a KlementievRothWordFeatureGen.
  if (_addBias) {
    stringstream ss;
    ss << y << FeatureGenConstants::PART_SEP << BiasFeatureGen::kPrefix;
    const int fId = _alphabet->lookup(ss.str(), true);
    if (fId >= 0)
      sub_pair_counts[fId] = RealWeight(RealWeight::kOne);
  }
  
  // TODO: Add the optional "distance" feature described in Feb. 17, 2011
  // email from M.W. Chang
  
  // If we're gathering features, keep track of the maximum number of non-zero
  // entries in a feature vector.
  if (!_alphabet->isLocked()) {
    const size_t entries = sub_pair_counts.size();
    if (entries > _maxEntries)
      _maxEntries = entries;
  }
  
  assert(sub_pair_counts.size() > 0);  
  FeatureVector<RealWeight>* fv = 0;  
  if (_pool)
    fv = _pool->get(sub_pair_counts);
  else
    fv = new FeatureVector<RealWeight>(sub_pair_counts);
  assert(fv);

  if (_normalize) {
    double normalization = x.getSize();
    assert(normalization > 0);
    if (_legacy) {
      // In the old code, we divided by the length of the longest string, where
      // the length was determined after padding the string with begin/end of
      // word markers. In the new code, that procedure would yield different
      // normalizers in the KR and bias feature gens, which is undesirable.
      normalization += 2;
    }
    fv->timesEquals(1.0 / normalization);
  }

  return fv;
}

inline void KlementievRothWordFeatureGen::appendSubstrings(const vector<string>* s,
    size_t i, size_t k, size_t end, vector<string>& subs) {
  string sub = s->at(i);
  subs.push_back(sub);
  for (size_t j = 1; j < k && i + j < end; j++) {
    sub = sub + CHAR_JOINER + s->at(i + j);
    subs.push_back(sub);
  }
}
