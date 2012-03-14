/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentPart.h"
#include "EditOperation.h"
#include "FeatureGenConstants.h"
#include "FeatureVector.h"
#include "Label.h"
#include "OpNone.h"
#include "Pattern.h"
#include "StateType.h"
#include "StringPair.h"
#include "WordAlignmentFeatureGen.h"
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace boost;
using namespace std;

WordAlignmentFeatureGen::WordAlignmentFeatureGen(
    shared_ptr<Alphabet> alphabet, int order, bool includeStateNgrams,
      bool includeAlignNgrams, bool includeCollapsedAlignNgrams,
      bool includeOpFeature, bool normalize) :
    AlignmentFeatureGen(alphabet), _order(order),
        _includeStateNgrams(includeStateNgrams),
        _includeAlignNgrams(includeAlignNgrams),
        _includeCollapsedAlignNgrams(includeCollapsedAlignNgrams),
        _includeOpFeature(includeOpFeature),
        _normalize(normalize), _addContextFeats(false) {
}

int WordAlignmentFeatureGen::processOptions(int argc, char** argv) {
  const string NONE = "None";
  stringstream vowelsHelp;
  vowelsHelp << "the name of a file whose first line contains a string of "
      << "vowels (case-insensitive), e.g., \"aeiou\" (sans quotes), (note: "
      << "this option activates consonant/vowel n-gram features); pass \""
      << NONE << "\" instead of a filename to disable";
  namespace opt = boost::program_options;
  bool noAlign = false;
  bool noCollapse = false;
  bool noNormalize = false;
  bool noOpFeature = false;
  bool noState = false;
  string vowelsFname;
  opt::options_description options(name() + " options");
  options.add_options()
    ("add-context-feats", opt::bool_switch(&_addContextFeats), "add features \
for combinations of the previous and next characters in the two strings")
    ("no-align-ngrams", opt::bool_switch(&noAlign), "do not include n-gram \
features of the aligned strings")
    ("no-collapsed-align-ngrams", opt::bool_switch(&noCollapse), "do not \
include backoff features of the aligned strings that discard the gaps/epsilons")
    ("no-normalize", opt::bool_switch(&noNormalize), "do not normalize by the \
length of the longer word")
    ("no-op-feature", opt::bool_switch(&noOpFeature), "do not include a \
feature that indicates the edit operation that was used")
    ("no-state-ngrams", opt::bool_switch(&noState), "do not include n-gram \
features of the state sequence")
    ("order", opt::value<int>(&_order), "the Markov order")
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
  
  if (noAlign)
    _includeAlignNgrams = false;
  if (noCollapse)
    _includeCollapsedAlignNgrams = false;
  if (noNormalize)
    _normalize = false;
  if (noOpFeature)
    _includeOpFeature = false;
  if (noState)
    _includeStateNgrams = false;

  if (vowelsFname != "" && !iequals(vowelsFname, NONE))
  {
    _vowelsRegex = "";
    ifstream fin(vowelsFname.c_str());
    if (!fin.good()) {
      cout << "Error: Unable to open " << vowelsFname << endl;
      return 1;
    }
    getline(fin, _vowelsRegex);
    fin.close();
    if (_vowelsRegex.size() == 0) {
      cout << "Error: The first line of the vowels file does not contain a "
          << "string\n";
      return 1;
    }
  }
  else
    _vowelsRegex = "aeiou"; // provide a default so regex doesn't choke below
  
  return 0;
}

FeatureVector<RealWeight>* WordAlignmentFeatureGen::getFeatures(
    const Pattern& x, Label label, int iNew, int jNew,
    const EditOperation& op, const vector<AlignmentPart>& history) {
  //const vector<string>& source = ((const StringPair&)x).getSource();
  //const vector<string>& target = ((const StringPair&)x).getTarget();
    
  const int histLen = history.size();
  assert(iNew >= 0 && jNew >= 0);
  assert(histLen >= 1);
  assert(_order >= 0);
  
  set<int> featureIds;
  const char* sep = FeatureGenConstants::PART_SEP;
  const bool includeVowels = _vowelsRegex.size() > 0;
  const string V = "[V]";
  const string C = "[C]";
  
  // The vowel regex matches any of the characters read from the vowels file.
  regex regVowel("["+_vowelsRegex+"]", regex::icase|regex::perl);
  
  // The consonant regex matches anything that's not a vowel, punctuation, or
  // a space.
  const string patt = "[^[:punct:]" + _vowelsRegex + "\\s]";
  regex regConsonant(patt, regex::icase|regex::perl);
  
  // For some features, we will want to remove any separators from phrases.
  const string escape = "\\";
  regex regPhraseSep(escape + FeatureGenConstants::PHRASE_SEP);
  
  // Determine the point in the history where the longest n-gram begins.
  typedef vector<AlignmentPart>::const_iterator align_iterator;
  int left;
  if (_order + 1 > histLen)
    left = 0;
  else
    left = histLen - (_order + 1);
  assert(left >= 0);

  // TODO: See if FastFormat (or some other int2str method) improves efficiency
  // http://stackoverflow.com/questions/191757/c-concatenate-string-and-int
  
  // Extract n-grams of the state sequence (up to the Markov order).
  if (_includeStateNgrams) {
    stringstream prefix;
    prefix << label << sep << "S:";
    string s;
    for (int k = histLen - 1; k >= left; k--) {
      s = history[k].state.getName() + s;
      addFeatureId(prefix.str() + s, featureIds);
      s = FeatureGenConstants::OP_SEP + s;
    }
  }

  // These features only fire if we didn't perform a no-op on this arc.
  if (op.getId() != OpNone::ID) {
    stringstream prefix;
    prefix << label << sep << "A:";
    
    if (_includeAlignNgrams) {
      string s;
      for (int k = histLen - 1; k >= left; k--) {
        s = history[k].source + FeatureGenConstants::OP_SEP +
            history[k].target + s;
        addFeatureId(prefix.str() + s, featureIds);
        if (includeVowels) {
          const string temp = regex_replace(s, regConsonant, C);
          const string fVC = regex_replace(temp, regVowel, V);
          addFeatureId(prefix.str() + fVC, featureIds);
        }
        s = FeatureGenConstants::PART_SEP + s;
      }
    }
    
    if (_includeCollapsedAlignNgrams) {
      string s, t;
      for (int k = histLen - 1, l = 1; k >= left; k--, l++) {
        if (history[k].source != FeatureGenConstants::EPSILON)
          s = history[k].source + s;
        if (history[k].target != FeatureGenConstants::EPSILON)
          t = history[k].target + t;
        // Since collapsed n-grams of different sizes may not be unique, we
        // prepend the size/length of the n-gram to each feature.
        stringstream size;
        size << l << FeatureGenConstants::PART_SEP;
        string collapsed = s + FeatureGenConstants::OP_SEP + t;
        collapsed = regex_replace(collapsed, regPhraseSep, "");
        addFeatureId(prefix.str() + size.str() + collapsed, featureIds);
        if (includeVowels) {
          const string temp = regex_replace(collapsed, regConsonant, C);
          const string collapsedVC = regex_replace(temp, regVowel, V);
          addFeatureId(prefix.str() + size.str() + collapsedVC, featureIds);
        }
      }
    }
  }
  
  if (_includeOpFeature) {
    stringstream ss;
    ss << label << sep << "E:" << op.getName();
    addFeatureId(ss.str(), featureIds);
  }
  
  FeatureVector<RealWeight>* fv = new FeatureVector<RealWeight>(featureIds);
  assert(fv);

  if (_normalize) {
    const double normalization = x.getSize();
    assert(normalization > 0);
    fv->timesEquals(1.0 / normalization);
  }
  
  return fv;
}

double WordAlignmentFeatureGen::getDefaultFeatureWeight(const string& f) const {
  typedef tokenizer< char_separator<char> > Tokenizer;
  char_separator<char> delims("_:>"); // FIXME: These shouldn't be hard-coded
  Tokenizer tokens(f, delims);
  Tokenizer::const_iterator it = tokens.begin();
  
  // FIXME: These should be defined somewhere. Are they used by anything else?
  const string MATCH("1");
  const string MISMATCH("0");
  
  const string label = *it++;
  assert(label == MATCH || label == MISMATCH);
  const int sign = (label == MATCH) ? 1 : -1;
  
  // FIXME: This function couples WordAlignmentFeatureGen with
  // StringEditModel, where the op names are set in processOptions().
  
  // Note: Unlike in the old version of the code, we don't need to consider
  // the generic and string-specific ops separately here because we defined
  // separate operations for identity and substitution (phrases differ in the
  // latter).
  
  const string opType = *it++;
  if (opType == "E") {                 // Edit (generic or string-specific)
    const string opName = *it++;
    if (istarts_with(opName, "No"))    // No-op
      return 0.0;
    if (istarts_with(opName, "Mat"))   // Match
      return sign;
    if (istarts_with(opName, "Sub"))   // Substitute
      return -sign;
    if (istarts_with(opName, "Del"))   // Delete
      return -sign;
    if (istarts_with(opName, "Ins"))   // Insert
      return -sign;
    if (istarts_with(opName, "Rep")) { // Replace
      if (it.at_end()) {
        // generic op: we don't know whether the strings differ; however, we
        // assume that more often than not, the strings will differ in practice,
        // so treat this as a Substitution
        return -sign;
      }
      const string source = *it++;
      const string target = *it++;
      if (source == target)
        return sign;
      else
        return -sign;
    }
    assert(0);
  }
  else if (opType == "S") {          // State transition feature
    double value = 0;
    const double a = 1.0/3.0;
    const double b = 1.0 - a;
    while (!it.at_end()) {
      // The most recent (rightmost) state should dominate (carry more weight),
      // so we interpolate the sign with a fraction of the previous value.
      const string stName = *it++;
      if (istarts_with(stName, "Mat"))
        value = a*value + b*sign; // A match is good
      else
        value = a*value - b*sign; // Everything else is (assumed to be) bad
    }
    return value;
  }
  else if (opType == "Bias") {       // bias (offset) feature
    // FIXME: This is actually an observed feature, but we're handling it here
    // for the sake of convenience.
    return sign;
  }
  //cout << "Warning: No default weight for: " << f << " (setting to zero)\n";
  return 0.0;
}

inline void WordAlignmentFeatureGen::addFeatureId(const string& f,
    set<int>& featureIds) const {
  const int fId = _alphabet->lookup(f, true);
  if (fId == -1)
    return;
  featureIds.insert(fId);
}

inline string WordAlignmentFeatureGen::extractPhrase(const vector<string>& str,
    int first, int last) {
  assert(last >= first);
  assert(first >= 0);
  const int size = str.size();
  if (last > size)
    last = size;
  if (last == first)
    return "";
  if (first+1 == last)
    return str[first];
  stringstream ss;
  for (int i = first; i < last-1; i++)
    ss << str[i] << FeatureGenConstants::PHRASE_SEP;
  ss << str[last-1];
  return ss.str();
}
