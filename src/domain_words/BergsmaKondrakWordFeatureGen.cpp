/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Alphabet.h"
#include "BergsmaKondrakWordFeatureGen.h"
#include "BiasFeatureGen.h"
#include "FeatureGenConstants.h"
#include "Label.h"
#include "StringPairAligned.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>
#include <boost/unordered_map.hpp>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

const string BergsmaKondrakWordFeatureGen::CHAR_JOINER = "+";
const string BergsmaKondrakWordFeatureGen::SUB_JOINER = ",";
const string BergsmaKondrakWordFeatureGen::MISMATCH_PREFIX = "MM:";

BergsmaKondrakWordFeatureGen::BergsmaKondrakWordFeatureGen(
    boost::shared_ptr<Alphabet> alphabet, bool normalize) :
    ObservedFeatureGen(alphabet), _substringSize(2), _normalize(normalize),
    _addMismatches(true), _addBias(true) {
}

int BergsmaKondrakWordFeatureGen::processOptions(int argc, char** argv) {
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
  bool noMismatches = false;
  options.add_options()
    ("bk-no-bias", opt::bool_switch(&noBias), "do not add a bias feature")
    ("bk-no-mismatches", opt::bool_switch(&noMismatches),
        "do not include mismatch features")
    ("bk-no-normalize", opt::bool_switch(&noNormalize),
        "do not normalize by the length of the longer word")
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
  
  if (noBias)
    _addBias = false;
  if (noMismatches)
    _addMismatches = false;
  if (noNormalize)
    _normalize = false;

  return 0;
}

bool containsMatch(const deque<int>& win, const bool* match) {
  BOOST_FOREACH(int index, win) {
    if (match[index])
      return true;
  }
  return false;
}

bool matchesAreConsistent(const vector<string>& s, const vector<string>& t,
    const deque<int>& sw, const deque<int>& tw, const bool* match) {
  BOOST_FOREACH(int index, sw) {
    // If there's a match at an index in the source window, but this index is
    // not in the target window, then we don't have a valid phrase pair.
    if (match[index] && find(tw.begin(), tw.end(), index) == tw.end())
      return false;
  }
  BOOST_FOREACH(int index, tw) {
    // Vice versa from the above.
    if (match[index] && find(sw.begin(), sw.end(), index) == sw.end())
      return false;
  }
  return true;
}

void BergsmaKondrakWordFeatureGen::getPhrasePairs(const vector<string>& s,
    const vector<string>& t, int sk, int tk, unordered_map<int, int>& fv,
    const bool* match, const Label y) {
  assert(s.size() == t.size());
  const char* EPS = FeatureGenConstants::EPSILON; // convenient short-hand
  const int length = s.size();
  
  for (int pos = 0; pos < length; pos++) {
    if (s[pos] == EPS)
      continue; // there is no phrase at this position in the source
    deque<int> s_win; // source window (stores indices)
    int i = pos;
    int j = -1;
    
    // Determine the initial source window.
    while (s_win.size() < sk && i < length) {
      if (s[i] != EPS) {
        s_win.push_back(i);
        if (match[i] and j < 0)
          j = i;
      }
      i++;
    }
    
    // We don't have a window of the desired width at this position.
    if (s_win.size() < sk)
      continue;
      
    // There are no matches in the source window, so no phrase pairs.
    if (j < 0)
      continue;
      
    // Grow the initial target window left from the first match in s_win.
    deque<int> t_win; // target window (stores indices)
    while (t_win.size() < tk and j >= 0) {
      if (t[j] != EPS)
        t_win.push_front(j);
      j--;
    }
    
    // Grow the target window up to width tk.
    j = t_win.back() + 1;
    while (t_win.size() < tk && j < length) {
      if (t[j] != EPS)
        t_win.push_back(j);
      j++;
    }
    
    // Add a phrase pair for the initial source and target windows, provided
    // that the criteria for a valid pair are satisfied.
    if (matchesAreConsistent(s, t, s_win, t_win, match))
      appendPhrasePair(s, t, s_win, t_win, fv, y);
      
    // Slide the target window to the right and extract additional phrases.
    j = t_win.back() + 1;
    while (j < length) {
      // If there is a match at position j, but j is not in the source window,
      // then there are no more phrase pairs for this source window.
      if (match[j] && find(s_win.begin(), s_win.end(), j) == s_win.end())
        break;
      if (t[j] != EPS) {
        t_win.push_back(j);
        t_win.pop_front();
        if (containsMatch(t_win, match) && matchesAreConsistent(s, t, s_win,
            t_win, match)) {
          appendPhrasePair(s, t, s_win, t_win, fv, y);
        }
      }
      j++;
    }
  }
}

SparseRealVec* BergsmaKondrakWordFeatureGen::getFeatures(const Pattern& x,
    const Label y) {
  // The strings have to be aligned (StringPairAligned instead of StringPair).
  const StringPairAligned& pair = (const StringPairAligned&)x;
  
  const vector<string>& s = pair.getSource();
  const vector<string>& t = pair.getTarget();
  assert(s.size() > 0 && t.size() > 0);
  assert(s.size() == t.size());
  
  unordered_map<int, int> phrase_pair_counts; // phrase pair counts
  
  // This array tells you, for a given position, whether or not there is a
  // match (i.e., the source and target characters are aligned/linked).
  scoped_array<bool> matched(new bool[s.size()]);
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == t[i]) {
      // We should never have an epsilon aligned to an epsilon.
      assert(s[i] != FeatureGenConstants::EPSILON);
      matched[i] = true;
    }
    else
      matched[i] = false;
  }
  
  for (int sk = 1; sk <= _substringSize; sk++)
    for (int tk = 1; tk <= _substringSize; tk++)
      getPhrasePairs(s, t, sk, tk, phrase_pair_counts, matched.get(), y);
  
  if (_addMismatches)
    getMismatches(s, t, phrase_pair_counts, y);
  
  // Add a bias feature if this option is enabled.
  // TODO: In the future, it would be cleaner to be able to activate multiple
  // feature generators via command line options. Here, we essentially have
  // a BiasFeatureGen inside of a BergsmaKondrakWordFeatureGen.
  if (_addBias) {
    const int fId = _alphabet->lookup(BiasFeatureGen::kPrefix, y, true);
    if (fId >= 0)
      phrase_pair_counts[fId] = 1;
  }
  
  const size_t d = _alphabet->size();
  SparseRealVec* fv = new SparseRealVec(_alphabet->size());
  
  unordered_map<int, int>::const_iterator it;
  for (it = phrase_pair_counts.begin(); it != phrase_pair_counts.end(); ++it) {
    const size_t index = it->first;
    if (index >= d) {
      // Assume we're in feature gathering mode, in which case the fv we return
      // will be ignored.
      return fv;
    }
    (*fv)(index) = it->second;
  }
  
  if (_normalize) {
    const double normalization = x.getSize();
    assert(normalization > 0);
    (*fv) /= normalization;
  }

  return fv;
}

void BergsmaKondrakWordFeatureGen::appendPhrasePair(const vector<string>& s,
    const vector<string>& t, const deque<int>& sw, const deque<int>& tw,
    unordered_map<int, int>& fv, const Label y) {
  stringstream phrasePair;
  int count = 0;
  BOOST_FOREACH(int i, sw) {
    phrasePair << s[i];
    if (count++)
      phrasePair << CHAR_JOINER;
  }
  phrasePair << SUB_JOINER;
  count = 0;
  BOOST_FOREACH(int j, tw) {
    phrasePair << t[j];
    if (count++)
      phrasePair << CHAR_JOINER;
  }
  // At test time, the alphabet will presumably be locked, and we don't
  // want to count unseen features; so, we pretend we never saw them.
  int fId = _alphabet->lookup(phrasePair.str(), y, true);
  if (fId >= 0)
    fv[fId]++;
}

// Ported from characterPairs.cpp in Shane Bergsma's implementation.
// See http://webdocs.cs.ualberta.ca/~bergsma/Cognates/
void BergsmaKondrakWordFeatureGen::getMismatches(const vector<string>& s,
    const vector<string>& t, unordered_map<int, int>& fv, const Label y) {
  assert(s.size() == t.size());

  string prevCharS = s[0];
  string prevCharT = t[0];
  assert(prevCharS == prevCharT); // the first characters (^) must always match
  bool prevMatch = true;

  for (int i = 0; i < s.size(); i++) {
    bool currMatch = s[i] == t[i];
    if (!currMatch) {
      prevCharS += s[i];
      prevCharT += t[i];
    }
    else {
      if (prevMatch) {
        prevCharS = s[i];
        prevCharT = t[i];
      }
      else {
        // Found a mismatch, so add the corresponding feature.
        prevCharS += s[i];
        prevCharT += t[i];
        {
          stringstream mismatch;
          mismatch << MISMATCH_PREFIX << prevCharS << SUB_JOINER << prevCharT;
          int fId = _alphabet->lookup(mismatch.str(), y, true);
          if (fId >= 0)
            fv[fId]++;
        }
        
        // Also create a mismatch feature without the left and right context.
        string shortS = prevCharS.substr(1, prevCharS.length() - 2);
        string shortT = prevCharT.substr(1, prevCharT.length() - 2);
        {
          stringstream mismatch;
          mismatch << MISMATCH_PREFIX << shortS << SUB_JOINER << SUB_JOINER
              << shortT;
          int fId = _alphabet->lookup(mismatch.str(), y, true);
          if (fId >= 0)
            fv[fId]++;
        }
        prevCharS = s[i];
        prevCharT = t[i];
      }
    }
    prevMatch = currMatch;
  }
}
