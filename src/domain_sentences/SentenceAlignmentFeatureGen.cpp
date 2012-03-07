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
#include "Pattern.h"
#include "SentenceAlignmentFeatureGen.h"
#include "StateType.h"
#include "StringPair.h"
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <list>
#include <sstream>
#include <string>
#include <vector>
using namespace boost;
using namespace std;

SentenceAlignmentFeatureGen::SentenceAlignmentFeatureGen(
    shared_ptr<Alphabet> alphabet, int order, bool includeAnnotatedEdits,
    bool includeEditFeats, bool includeStateNgrams, bool normalize) :
    AlignmentFeatureGen(alphabet), _order(order), _includeAnnotatedEdits(
        includeAnnotatedEdits), _includeEditFeats(includeEditFeats),
        _includeStateNgrams(includeStateNgrams), _normalize(normalize) {
}

int SentenceAlignmentFeatureGen::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  bool noAnnotated = false;
  bool noEdit = false;
  bool noState = false;
  bool noNormalize = false;
  opt::options_description options(name() + " options");
  options.add_options()
    ("no-annotate", opt::bool_switch(&noAnnotated), "do not include annotated \
edit operation features (e.g., edits along with affected characters)")
    ("no-edit", opt::bool_switch(&noEdit), "do not include edit operation \
features (overrides --no-annotate)")
    ("no-normalize", opt::bool_switch(&noNormalize), "do not normalize by the \
length of the longer sentence")
    ("no-state", opt::bool_switch(&noState), "do not include state \
(transition) features")
    ("order", opt::value<int>(&_order), "the Markov order")
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
  
  if (noAnnotated)
    _includeAnnotatedEdits = false;
  if (noEdit)
    _includeEditFeats = false;
  if (noNormalize)
    _normalize = false;
  if (noState)
    _includeStateNgrams = false;
  
  return 0;
}

FeatureVector<RealWeight>* SentenceAlignmentFeatureGen::getFeatures(
    const Pattern& x, Label label, int iNew, int jNew,
    const EditOperation& op, const vector<AlignmentPart>& stateHistory) {
  const vector<string>& source = ((const StringPair&)x).getSource();
  const vector<string>& target = ((const StringPair&)x).getTarget();
    
  // TODO: May want to do the tokenizing in the reader, store in a
  // different data structure (not StringPair) possibly with
  // feature ids for the "basic" features, which can then be
  // combined into "complex" features here in the FeatureGen.
  // Note that this would require making AlignmentTransduceconst vector<string>& sourcer more
  // generic, since it currently assumes two vectors of strings;
  // however, I think it only actually needs to know the "size" of
  // the source and target sequences.
    
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
  if (_includeStateNgrams) {
    ss.str(""); // re-initialize the stringstream
    ss << label << sep << "S:";
    int start;
    if (_order + 1 > histLen)
      start = 0;
    else
      start = histLen - (_order + 1);
    for (int k = start; k < histLen-1; k++) {
      ss << stateHistory[k].state.getName();
      addFeatureId(ss.str(), featureIds);
      ss << FeatureGenConstants::OP_SEP;
    }
    ss << stateHistory[histLen-1].state.getName();
    addFeatureId(ss.str(), featureIds);
  }

  // edit operation feature (state, operation interchangable in this function)
  if (_includeEditFeats && op.getId() != EditOperation::noopId()) {
    if (_includeAnnotatedEdits) {
      vector<string>::const_iterator sourcePhraseBegin, sourcePhraseEnd;
      const bool gotSource = false; // FIXME = getPhraseIterators(source, i, iNew,
          //sourcePhraseBegin, sourcePhraseEnd);
      vector<string>::const_iterator targetPhraseBegin, targetPhraseEnd;
      const bool gotTarget = false; // FIXME = getPhraseIterators(target, j, jNew,
          //targetPhraseBegin, targetPhraseEnd);
          
      typedef tokenizer<char_separator<char> > Tokenizer;
      char_separator<char> featSep(FeatureGenConstants::WORDFEAT_SEP);
            
      // Figure out how many indicator features there are. We assume that the
      // first feature name is "WRD" -- the id of the word itself. Since this
      // is a very sparse feature, we choose to ignore it below. However, it
      // could be used in the future to, e.g., lookup the word in some sort of
      // table to get a quantity that's of interest.
      int numFeats = 0;
      if (gotSource) {
        Tokenizer tokens(*sourcePhraseBegin, featSep);
        Tokenizer::const_iterator it = tokens.begin();
        assert(istarts_with(*it, "WRD")); // FIXME: Should not be hard-coded.
        for (; it != tokens.end(); ++it)
          ++numFeats;
      }
      else {
        assert(gotTarget);
        Tokenizer tokens(*targetPhraseBegin, featSep);
        Tokenizer::const_iterator it = tokens.begin();
        assert(istarts_with(*it, "WRD")); // FIXME: Should not be hard-coded.
        for (; it != tokens.end(); ++it)
          ++numFeats;
      }      
      --numFeats; // ignore WRD
      assert(numFeats > 0);
      
      vector<string> sourceFeats(numFeats);
      vector<string> targetFeats(numFeats);
      if (gotSource) {
        vector<string>::const_iterator phraseIt = sourcePhraseBegin;
        for (; phraseIt != sourcePhraseEnd; phraseIt++) {
          Tokenizer tokens(*phraseIt, featSep);
          Tokenizer::const_iterator it = tokens.begin();
          ++it; // skip WRD
          for (int fi = 0; fi < numFeats; fi++)
            sourceFeats[fi] += (*it++) + FeatureGenConstants::PHRASE_SEP;
          assert(it == tokens.end());
        }
      }
      if (gotTarget) {
        vector<string>::const_iterator phraseIt = targetPhraseBegin;
        for (; phraseIt != targetPhraseEnd; phraseIt++) {
          Tokenizer tokens(*phraseIt, featSep);
          Tokenizer::const_iterator it = tokens.begin();
          ++it; // skip WRD
          for (int fi = 0; fi < numFeats; fi++)
            targetFeats[fi] += (*it++) + FeatureGenConstants::PHRASE_SEP;
          assert(it == tokens.end());
        }
      }
        
      for (int fi = 0; fi < numFeats; fi++) {
        ss.str(""); // re-initialize the stringstream
        ss << label << sep << "E:" << op.getName() << ":" << sourceFeats[fi] <<
            FeatureGenConstants::OP_SEP << targetFeats[fi];
        addFeatureId(ss.str(), featureIds);
      }
    }
    ss.str(""); // re-initialize the stringstream
    ss << label << sep << "E:" << op.getName();
    addFeatureId(ss.str(), featureIds);
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

inline double SentenceAlignmentFeatureGen::getDefaultFeatureWeight(
    const string& f) const {
  return 0.0;
}

inline void SentenceAlignmentFeatureGen::addFeatureId(const string& f,
    list<int>& featureIds) const {
  const int fId = _alphabet->lookup(f, true);
  if (fId == -1)
    return;
  featureIds.push_back(fId);
}

inline bool SentenceAlignmentFeatureGen::getPhraseIterators(
    const vector<string>& str, int first, int last,
    vector<string>::const_iterator& itBegin,
    vector<string>::const_iterator& itEnd) {
  assert(last >= first);
  if (last > (int)str.size())
    last = str.size();
  if (last == first)
    return false;
  if (first+1 == last) {
    itBegin = str.begin() + first;
    itEnd = itBegin + 1;
    return true;
  }
  itBegin = str.begin() + first;
  itEnd = str.begin() + last;
  return true;
}
