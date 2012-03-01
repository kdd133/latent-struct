/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "FeatureGenConstants.h"
#include "FeatureVector.h"
#include "KlementievRothWordFeatureGen.h"
#include "KlementievRothSentenceFeatureGen.h"
#include "Label.h"
#include "ObservedFeatureGen.h"
#include "RealWeight.h"
#include "StringPair.h"
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <sstream>
#include <string>
#include <tr1/unordered_map>
#include <vector>
using namespace boost;
using namespace std;


KlementievRothSentenceFeatureGen::KlementievRothSentenceFeatureGen(
    boost::shared_ptr<Alphabet> alphabet, bool normalize) :
    ObservedFeatureGen(alphabet), _substringSize(2), _offsetSize(1),
    _normalize(normalize), _addBias(true) {
}

int KlementievRothSentenceFeatureGen::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  bool noBias = false;
  bool noNormalize = false;
  options.add_options()
    ("kr-no-bias", opt::bool_switch(&noBias), "do not add a bias feature")
    ("kr-no-normalize", opt::bool_switch(&noNormalize),
        "do not normalize by the length of the longer sentence")
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

FeatureVector<RealWeight>* KlementievRothSentenceFeatureGen::getFeatures(
    const Pattern& x, const Label y) {
  const StringPair& pair = (const StringPair&)x;
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

  char_separator<char> featSep(FeatureGenConstants::WORDFEAT_SEP);

  tokenizer<char_separator<char> > tokens(*shortest->begin(), featSep);
  tokenizer<char_separator<char> >::const_iterator iter = tokens.begin();
  assert(istarts_with(*iter, "WRD")); // FIXME: Should not be hard-coded.
  size_t nfeats = 0;
  for (; iter != tokens.end(); ++iter)
    ++nfeats;
  assert(nfeats > 0);

  for (size_t fi = 1; fi < nfeats; fi++) { // Note: fi = 1 --> skip WRD
    // extract the k-grams at each position in the shortest string
    for (int i = 0; i < (int) shortest->size(); i++) {
      subs_sh.clear();
      appendFeature(fi, shortest, i, _substringSize, shortest->size(),
          subs_sh, featSep);
      subs_lo.clear();

      // extract the k-grams at positions in the longest string that are within
      // _offsetSize of the current position in the shortest string
      for (int j = -_offsetSize; j <= _offsetSize; j++) {
        if (i + j >= 0 && i + j < (int) longest->size())
          appendFeature(fi, longest, i + j, _substringSize, longest->size(),
              subs_lo, featSep);
      }

      // pair each source substring with each target substring
      for (vector<string>::const_iterator subs_source_it = subs_source->begin();
          subs_source_it != subs_source->end(); subs_source_it++)
        for (vector<string>::const_iterator subs_target_it =
            subs_target->begin(); subs_target_it != subs_target->end();
            subs_target_it++) {
          stringstream ss;
          ss << y << FeatureGenConstants::PART_SEP << *subs_source_it <<
              KlementievRothWordFeatureGen::SUB_JOINER << *subs_target_it;       
          // at test time, the dictionary will be locked, and we don't want
          // to count unseen features; we pretend we never saw them
          const int fId = _alphabet->lookup(ss.str(), true);
          if (fId >= 0)
            sub_pair_counts[fId].plusEquals(RealWeight::kOne);
        }
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
    fv->timesEquals(1.0 / normalization);
  }

  return fv;
}

inline void KlementievRothSentenceFeatureGen::appendFeature(const int fi,
    const vector<string>* s, size_t i, size_t k, size_t end,
    vector<string>& subs, const char_separator<char>& featSep) {
  assert(fi >= 0);
  typedef tokenizer<char_separator<char> > Tokenizer;
  Tokenizer tokens(s->at(i), featSep);
  Tokenizer::const_iterator it = tokens.begin();
  for (int c = 0; c < fi; ++c)
    ++it;  
  string sub = *it;
  subs.push_back(sub);  
  for (size_t j = 1; j < k && i + j < end; j++) {
    Tokenizer tokens2(s->at(i + j), featSep);
    it = tokens2.begin();
    for (int c = 0; c < fi; ++c)
      ++it;
    sub = sub + KlementievRothWordFeatureGen::CHAR_JOINER + *it;
    subs.push_back(sub);
  }
}
