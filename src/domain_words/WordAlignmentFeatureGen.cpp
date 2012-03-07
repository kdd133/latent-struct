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
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace boost;
using namespace std;

WordAlignmentFeatureGen::WordAlignmentFeatureGen(
    shared_ptr<Alphabet> alphabet, int order, bool includeAnnotatedEdits,
    bool includeEditFeats, bool includeStateNgrams, bool normalize) :
    AlignmentFeatureGen(alphabet), _order(order), _includeAnnotatedEdits(
        includeAnnotatedEdits), _includeEditFeats(includeEditFeats),
        _includeStateNgrams(includeStateNgrams), _normalize(normalize),
        _addContextFeats(false), _legacy(false) {
}

int WordAlignmentFeatureGen::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  bool noAlign = false;
  bool noAnnotated = false;
  bool noEdit = false;
  bool noState = false;
  bool noNormalize = false;
  string vowelsFname;
  opt::options_description options(name() + " options");
  options.add_options()
    ("add-context-feats", opt::bool_switch(&_addContextFeats), "add features \
for combinations of the previous and next characters in the two strings")
    ("legacy", opt::bool_switch(&_legacy),
        "handle matching and mismatching phrases differently, as in old code")
    ("no-align-ngrams", opt::bool_switch(&noAlign), "do not include n-gram \
features of the aligned strings")
    ("no-annotate", opt::bool_switch(&noAnnotated), "do not include annotated \
edit operation features (e.g., edits along with affected characters)")
    ("no-edit", opt::bool_switch(&noEdit), "do not include edit operation \
features (overrides --no-annotate)")
    ("no-normalize", opt::bool_switch(&noNormalize), "do not normalize by the \
length of the longer word")
    ("no-state-ngrams", opt::bool_switch(&noState), "do not include n-gram \
features of the state sequence")
    ("order", opt::value<int>(&_order), "the Markov order")
    ("vowels-file", opt::value<string>(&vowelsFname), "the name of a file that \
contains a list of vowels, one per line")
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
  if (noAnnotated)
    _includeAnnotatedEdits = false;
  if (noEdit)
    _includeEditFeats = false;
  if (noNormalize)
    _normalize = false;
  if (noState)
    _includeStateNgrams = false;
    
  if (vowelsFname != "") {
    ifstream fin(vowelsFname.c_str());
    if (!fin.good()) {
      cout << "Error: Unable to open " << vowelsFname << endl;
      return 1;
    }
    string line;
    while (getline(fin, line)) {
      if (line.size() > 0)
        _vowels.insert(line);
    }
  }
  
  return 0;
}

FeatureVector<RealWeight>* WordAlignmentFeatureGen::getFeatures(
    const Pattern& x, Label label, int iNew, int jNew,
    const EditOperation& op, const vector<AlignmentPart>& stateHistory) {
  const vector<string>& source = ((const StringPair&)x).getSource();
  const vector<string>& target = ((const StringPair&)x).getTarget();
    
  const int histLen = stateHistory.size();
  assert(iNew >= 0 && jNew >= 0);
  assert(histLen >= 1);
  assert(_order >= 0);
  
  list<int> featureIds;
  stringstream ss;
  
  const char* sep = FeatureGenConstants::PART_SEP;

  // TODO: See if FastFormat (or some other int2str method) improves efficiency
  // http://stackoverflow.com/questions/191757/c-concatenate-string-and-int
  
  // n-grams of the state sequence (only valid up to the Markov order)
  cout << "in WordAlignment\n";
  if (_includeStateNgrams) {
    ss.str(""); // re-initialize the stringstream
    ss << label << sep << "S:";
    int start;
    if (_order + 1 > histLen)
      start = 0;
    else
      start = histLen - (_order + 1);
    // FIXME: This is buggy! We're building the n-gram from the left instead of
    // from the right (same thing in SentenceAlignmentFeatureGen.cpp)
    for (int k = start; k < histLen-1; k++) {
      ss << stateHistory[k].state.getName();
      addFeatureId(ss.str(), featureIds);
      ss << FeatureGenConstants::OP_SEP;
    }
    ss << stateHistory[histLen-1].state.getName();
    addFeatureId(ss.str(), featureIds);
  }

  if (_includeAlignNgrams) {
    ss.str(""); // re-initialize the stringstream
    assert(0);
    // TODO: It looks like I'll have to "reconstruct" the alignment based on
    // the state history, which means this function will have to know the inner
    // workings of each possible edit operation (?).
    // Maybe a better approach would be to maintain the current alignment string
    // (i.e., with epsilons) in the AlignmentTransducer function that calls this
    // one. It shouldn't be much harder than maintaining the state history,
    // which we already do, right?
    // Idea: Could pass in aligned strings as the StringPair/Patter (1st arg) to
    // this function. I don't think we would ever need the unaligned/original
    // strings. Could just remove the alignment info anyways, if the need were
    // to arise.
  }

  // edit operation feature (state, operation interchangable in this function)
  if (_includeEditFeats && op.getId() != OpNone::ID) {
    assert(!_legacy || _includeAnnotatedEdits);
    if (_includeAnnotatedEdits) {
      ss.str(""); // re-initialize the stringstream
      string sourcePhrase; // FIXME = extractPhrase(source, i, iNew);
      string targetPhrase; // FIXME = extractPhrase(target, j, jNew);
      ss << label << sep << "E:" << op.getName() << ":" << sourcePhrase
          << FeatureGenConstants::OP_SEP << targetPhrase;
      addFeatureId(ss.str(), featureIds);
      if (_legacy) {
        // For substitution-like edits (which consume characters from both the
        // source and target strings), fire the special feature SubDiff if the
        // phrases differ; otherwise, fire the usual op.getName() feature.
        ss.str(""); // re-initialize the stringstream
        if (sourcePhrase.size() > 0 && targetPhrase.size() > 0 &&
            sourcePhrase != targetPhrase)
          ss << label << sep << "E:SubDiff";
        else
          ss << label << sep << "E:" << op.getName();
        addFeatureId(ss.str(), featureIds);
      }
      if (_addContextFeats) { // FIXME
/*
        string sourceLeft;
        if (i <= 0)
          sourceLeft = FeatureGenConstants::BEGIN_CHAR;
        else
          sourceLeft = source[i - 1];
        string sourceRight;
        if (iNew >= (int)source.size())
          sourceRight = FeatureGenConstants::END_CHAR;
        else
          sourceRight = source[iNew];
        string targetLeft;
        if (j <= 0)
          targetLeft = FeatureGenConstants::BEGIN_CHAR;
        else
          targetLeft = target[j - 1];
        string targetRight;
        if (jNew >= (int)target.size())
          targetRight = FeatureGenConstants::END_CHAR;
        else
          targetRight = target[jNew];
          
        ss.str("");
        ss << label << sep << "E:" << op.getName() << ":" << sourcePhrase
          << FeatureGenConstants::OP_SEP << targetPhrase << sep << "sl="
          << sourceLeft;
        addFeatureId(ss.str(), featureIds);
        
        ss.str("");
        ss << label << sep << "E:" << op.getName() << ":" << sourcePhrase
          << FeatureGenConstants::OP_SEP << targetPhrase << sep << "sr="
          << sourceRight;
        addFeatureId(ss.str(), featureIds);
        
        ss.str("");
        ss << label << sep << "E:" << op.getName() << ":" << sourcePhrase
          << FeatureGenConstants::OP_SEP << targetPhrase << sep << "tl="
          << targetLeft;
        addFeatureId(ss.str(), featureIds);
        
        ss.str("");
        ss << label << sep << "E:" << op.getName() << ":" << sourcePhrase
          << FeatureGenConstants::OP_SEP << targetPhrase << sep << "tr="
          << targetRight;
        addFeatureId(ss.str(), featureIds);
*/
      }
    }
    if (!_legacy) {
      ss.str(""); // re-initialize the stringstream
      ss << label << sep << "E:" << op.getName();
      addFeatureId(ss.str(), featureIds);
    }
  }

  if (!_alphabet->isLocked()) {
    const size_t entries = featureIds.size();
    if (entries > _maxEntries)
      _maxEntries = entries;
  }
  
  FeatureVector<RealWeight>* fv = 0;
  if (_pool)
    fv = _pool->get(featureIds);
  else
    fv = new FeatureVector<RealWeight>(featureIds);
    
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
  
  // TODO: ANY CHANGES NEEDED HERE AFTER UPDATING FEATURE FUNCTION ABOVE?
  const string opType = *it++;
  if (opType == "E") {                 // Generic or string-specific op feature
    const string opName = *it++;
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
        return _legacy ? -sign : sign; // match (legacy had this backward)
      return _legacy ? sign : -sign; // substitution (legacy had this backward)
    }
    assert(0);
  }
  else if (opType == "Bias") {       // bias (offset) feature
    // FIXME: This is actually an observed feature, but we're handling it here
    // for the sake of convenience.
    return (label == MATCH) ? 1 : -1;
  }
  //cout << "Warning: No default weight for: " << f << " (setting to zero)\n";
  return 0.0;
}

inline void WordAlignmentFeatureGen::addFeatureId(const string& f,
    list<int>& featureIds) const {
  const int fId = _alphabet->lookup(f, true);
  if (fId == -1)
    return;
  featureIds.push_back(fId);
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
