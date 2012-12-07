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
#include "KlementievRothWordFeatureGen.h"
#include "Label.h"
#include "StringPair.h"
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/scoped_array.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <tr1/unordered_map>
#include <vector>

using namespace boost;
using namespace std;

const string KlementievRothWordFeatureGen::CHAR_JOINER = "+";
const string KlementievRothWordFeatureGen::SUB_JOINER = ",";

KlementievRothWordFeatureGen::KlementievRothWordFeatureGen(
    boost::shared_ptr<Alphabet> alphabet, bool normalize) :
    ObservedFeatureGen(alphabet), _substringSize(2), _offsetSize(1),
    _normalize(normalize), _addBias(true), _regexEnabled(false) {
}

int KlementievRothWordFeatureGen::processOptions(int argc, char** argv) {
  const string NONE = "None";
  stringstream vowelsHelp;
  vowelsHelp << "the name of a file whose first line contains a string of "
      << "vowels (case-insensitive), e.g., \"aeiou\" (sans quotes), (note: "
      << "this option activates consonant/vowel n-gram features); pass \""
      << NONE << "\" instead of a filename to disable";
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  bool noBias = false;
  bool noNormalize = false;
  string vowelsFname;
  options.add_options()
    ("kr-no-bias", opt::bool_switch(&noBias), "do not add a bias feature")
    ("kr-no-normalize", opt::bool_switch(&noNormalize),
        "do not normalize by the length of the longer word")
    ("offset-size", opt::value<int>(&_offsetSize)->default_value(1),
        "allow substring positions to differ by up to +- this number")
    ("substring-size", opt::value<int>(&_substringSize)->default_value(2),
        "extract substrings up to this length")
    ("vowels-file", opt::value<string>(&vowelsFname), vowelsHelp.str().c_str())
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

  if (vowelsFname != "" && !iequals(vowelsFname, NONE))
  {
    string vowelsRegexStr;
    ifstream fin(vowelsFname.c_str());
    if (!fin.good()) {
      cout << "Error: Unable to open " << vowelsFname << endl;
      return 1;
    }
    getline(fin, vowelsRegexStr);
    fin.close();
    if (vowelsRegexStr.size() == 0) {
      cout << "Error: The first line of the vowels file does not contain a "
          << "string\n";
      return 1;
    }
    _regexEnabled = true;
    
    // The vowel regex matches any of the characters read from the vowels file.
    _regVowel = regex("[" + vowelsRegexStr + "]", regex::icase|regex::perl);
    
    // The consonant regex matches anything that's not a vowel, punctuation, or
    // a space.
    string patt = "[^[:punct:]" + vowelsRegexStr + "\\s]";
    _regConsonant = regex(patt, regex::icase|regex::perl);
  }

  return 0;
}

SparseRealVec* KlementievRothWordFeatureGen::getFeatures(const Pattern& x,
    const Label y) {
  const StringPair& pair = (const StringPair&)x;
  
  // Vowel and consonant markers (used by regex_replace below).
  const string V = "[V]";
  const string C = "[C]";
  
  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  assert(s.size() > 0 && t.size() > 0);
  
  const vector<string>* longest; // pointer to the longest string
  const vector<string>* shortest; // pointer to the shortest string

  vector<string> subs_sh; // substrings from the shortest string
  vector<string> subs_lo; // substrings from the longest string
  vector<string>* subs_source; // pointer to the source substrings
  vector<string>* subs_target; // pointer to the target substrings
  
  scoped_array<int> sub_pair_counts(new int[_alphabet->size()]);
  for (size_t i = 0; i < _alphabet->size(); ++i)
    sub_pair_counts[i] = 0;

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
    
    stringstream prefix;
    prefix << y << FeatureGenConstants::PART_SEP;

    // pair each source substring with each target substring
    for (vector<string>::const_iterator subs_source_it = subs_source->begin();
        subs_source_it != subs_source->end(); subs_source_it++)
      for (vector<string>::const_iterator subs_target_it = subs_target->begin();
          subs_target_it != subs_target->end(); subs_target_it++) {
        stringstream phrasePair;
        phrasePair << *subs_source_it << SUB_JOINER << *subs_target_it;
        // At test time, the alphabet will presumably be locked, and we don't
        // want to count unseen features; so, we pretend we never saw them.
        int fId = _alphabet->lookup(prefix.str() + phrasePair.str(), true);
        if (fId >= 0)
          sub_pair_counts[fId]++;
          
        if (_regexEnabled) {
          const string temp = regex_replace(phrasePair.str(), _regConsonant, C);
          const string VC = regex_replace(temp, _regVowel, V);
          fId = _alphabet->lookup(prefix.str() + VC, true);
          if (fId >= 0)
            sub_pair_counts[fId]++;
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
      sub_pair_counts[fId] = 1;
  }
  
  // TODO: Add the optional "distance" feature described in Feb. 17, 2011
  // email from M.W. Chang
  
  SparseRealVec* fv = new SparseRealVec(_alphabet->size());
  for (size_t i = 0; i < _alphabet->size(); ++i) {
    if (sub_pair_counts[i] > 0)
      (*fv)(i) = sub_pair_counts[i];
  }

  if (_normalize) {
    const double normalization = x.getSize();
    assert(normalization > 0);
    (*fv) /= normalization;
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
