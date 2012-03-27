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

WordAlignmentFeatureGen::WordAlignmentFeatureGen(shared_ptr<Alphabet> alphabet)
    : AlignmentFeatureGen(alphabet), _order(1), _includeStateNgrams(true),
    _includeAlignNgrams(true), _includeCollapsedAlignNgrams(true),
    _includeBigramFeatures(false), _normalize(true), _regexEnabled(false),
    _alignUnigramsOnly(false) {
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
  bool noState = false;
  string vowelsFname;
  opt::options_description options(name() + " options");
  options.add_options()
    ("align-unigrams-only", opt::bool_switch(&_alignUnigramsOnly), "exclude \
higher order alignment n-grams, even if --order > 0")
    ("bigram-features", opt::bool_switch(&_includeBigramFeatures), "include a \
source-unigram to target-bigram feature (and vice versa), where each n-gram \
is extracted from the phrase pairs involved in the current edit operation")
    ("no-align-ngrams", opt::bool_switch(&noAlign), "do not include n-gram \
features of the aligned strings")
    ("no-collapsed-align-ngrams", opt::bool_switch(&noCollapse), "do not \
include backoff features of the aligned strings that discard the gaps/epsilons")
    ("no-normalize", opt::bool_switch(&noNormalize), "do not normalize by the \
length of the longer word")
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
  if (noState)
    _includeStateNgrams = false;

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
  
  if (string(FeatureGenConstants::PHRASE_SEP) != "+") {
    cout << "Error: The value of FeatureGenConstants::PHRASE_SEP is assumed to \
be '+', but it was changed. The regular expression(s) in WordAlignmentFeature\
Gen need to be updated!\n";
    return 1;
  }
  // For some features, we will want to remove redundant phrase separators.
  // These may be introduced when as we concatenate phrases in the if statement 
  // involving _includeCollapsedAlignNgrams in getFeatures() below.
  // The first regex matches 2 or more consecutive + signs; the second matches
  // a leading or trailing + sign.
  _regPhraseSepMulti = regex("\\+{2,}");
  _regPhraseSepLeadTrail = regex("(^\\+|\\+$)");
  
  return 0;
}

FeatureVector<RealWeight>* WordAlignmentFeatureGen::getFeatures(
    const Pattern& x, Label label, int sourcePos, int targetPos,
    const EditOperation& op, const vector<AlignmentPart>& history) {
    
  const int histLen = history.size();
  assert(sourcePos >= 0 && targetPos >= 0);
  assert(histLen >= 1);
  assert(_order >= 0);
  
  set<int> featureIds;
  const string sep = FeatureGenConstants::PART_SEP;
  const string V = "[V]";
  const string C = "[C]";
  
  // Determine the point in the history where the longest n-gram begins.
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
    for (int k = histLen - 1, n = 1; k >= left; k--, n++) {
      s = history[k].opName + s;
      addFeatureId(prefix.str() + s, featureIds);
      s = FeatureGenConstants::OP_SEP + s;
    }
  }

#if 0
  // Show the history.
  for (size_t k = 0; k < history.size(); k++)
    cout << "(" << history[k].source << ">" << history[k].target << " " <<
      history[k].opName << ")";
  cout << endl;
#endif
    
  // Note: We require histLen > 1 because histLen == 1 will only be true at the
  // start state, in which case no edit operation has yet been performed (i.e.,
  // the alignment feature would just be ->-; but, the state n-gram feature
  // above already implies this).
  if ((_includeAlignNgrams || _includeCollapsedAlignNgrams) && histLen > 1) {
    stringstream prefix;
    prefix << label << sep << "A:";
    string alignNgram;
    for (int k = histLen - 1, n = 1; k >= left; k--, n++) {
      alignNgram = history[k].source + FeatureGenConstants::OP_SEP +
          history[k].target + alignNgram;
      // The last condition in the if statements implies that we don't fire an
      // alignment unigram feature if the operation was a noop.
      if (_includeAlignNgrams && (!_alignUnigramsOnly || n == 1) &&
          (op.getId() != OpNone::ID || n > 1)) {
        addFeatureId(prefix.str() + alignNgram, featureIds);
        if (_regexEnabled) {
          const string temp = regex_replace(alignNgram, _regConsonant, C);
          const string alignNgramVC = regex_replace(temp, _regVowel, V);
          addFeatureId(prefix.str() + alignNgramVC, featureIds);
        }
      }
      if (k > left)
        alignNgram = FeatureGenConstants::PART_SEP + alignNgram;
    }
  }
  
  // Collapsed n-gram features discard epsilons, so that strings aligned as
  // (i,-)(e,e) and (-,i)(e,e) both produce the collapsed feature (ie,e).
  // Note that in the case of a zero order model, the collapsed features are
  // always redundant, so we omit them in this case.
  if (_includeCollapsedAlignNgrams && !_alignUnigramsOnly && _order > 0 &&
      histLen > 1) {
    stringstream prefix;
    prefix << label << sep << "C:";
    string s, t;
    for (int k = histLen - 1, n = 1; k >= left; k--, n++) {
      if (history[k].source != FeatureGenConstants::EPSILON)
        s = history[k].source + sep + s;
      if (history[k].target != FeatureGenConstants::EPSILON)
        t = history[k].target + sep + t;
        
      // Omit zero order feature (effectively a duplicate of the A: feature).
      if (_includeAlignNgrams && n == 1)
        continue;
        
      // Replace two or more consecutive seps with a single sep, then remove
      // any leading or trailing sep(s).
      string sStrip = regex_replace(s, _regPhraseSepMulti, sep);
      sStrip = regex_replace(sStrip, _regPhraseSepLeadTrail, "");
      string tStrip = regex_replace(t, _regPhraseSepMulti, sep);
      tStrip = regex_replace(tStrip, _regPhraseSepLeadTrail, "");
      string collapsed = sStrip + FeatureGenConstants::OP_SEP + tStrip;
      
      addFeatureId(prefix.str() + collapsed, featureIds);
      if (_regexEnabled) {
        const string temp = regex_replace(collapsed, _regConsonant, C);
        const string collapsedVC = regex_replace(temp, _regVowel, V);
        addFeatureId(prefix.str() + collapsedVC, featureIds);
      }
    }
  }
  
  if (_includeBigramFeatures) {
    // We may need to look at the source and/or target strings.
    const vector<string>& sourceSeq = ((const StringPair&)x).getSource();
    const vector<string>& targetSeq = ((const StringPair&)x).getTarget();
    
    const AlignmentPart& edit = history.back();
    string sourceBi;
    string targetBi;
    
    if (edit.source != FeatureGenConstants::EPSILON) {
      sourceBi = edit.source + FeatureGenConstants::PHRASE_SEP;
      if (sourcePos < sourceSeq.size())
        sourceBi += sourceSeq[sourcePos];
      else
        sourceBi += FeatureGenConstants::END_CHAR;
    }
    if (edit.target != FeatureGenConstants::EPSILON) {
      targetBi = edit.target + FeatureGenConstants::PHRASE_SEP;
      if (targetPos < targetSeq.size())
        targetBi += targetSeq[targetPos];
      else
        targetBi += FeatureGenConstants::END_CHAR;
    }
    
    if (edit.source != FeatureGenConstants::EPSILON) {
      stringstream f;
      f << label << sep << "Bi:" << sourceBi << FeatureGenConstants::OP_SEP
          << edit.target;
      addFeatureId(f.str(), featureIds);
    }    
    if (edit.target != FeatureGenConstants::EPSILON) {
      stringstream f;
      f << label << sep << "Bi:" << edit.source << FeatureGenConstants::OP_SEP
          << targetBi;
      addFeatureId(f.str(), featureIds);
    }
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
