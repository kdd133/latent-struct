/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentPart.h"
#include "Alphabet.h"
#include "EditOperation.h"
#include "FeatureGenConstants.h"
#include "LogWeight.h"
#include "OpNone.h"
#include "Pattern.h"
#include "SentenceAlignmentFeatureGen.h"
#include "StateType.h"
#include "StringPair.h"
#include "Ublas.h"
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tokenizer.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

SentenceAlignmentFeatureGen::SentenceAlignmentFeatureGen(
    shared_ptr<Alphabet> alphabet) : AlignmentFeatureGen(alphabet), _order(1),
    _includeStateNgrams(true), _includeAlignNgrams(true),
    _alignUnigramsOnly(false), _normalize(true) {
}

int SentenceAlignmentFeatureGen::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  bool noAlign = false;
  bool noNormalize = false;
  bool noState = false;
  opt::options_description options(name() + " options");
  options.add_options()
    ("align-unigrams-only", opt::bool_switch(&_alignUnigramsOnly), "exclude \
higher order alignment n-grams, even if --order > 0")
    ("no-align-ngrams", opt::bool_switch(&noAlign), "do not include n-gram \
features of the aligned strings")
    ("no-normalize", opt::bool_switch(&noNormalize), "do not normalize by the \
length of the longer word")
    ("no-state-ngrams", opt::bool_switch(&noState), "do not include n-gram \
features of the state sequence")
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
  
  if (noAlign)
    _includeAlignNgrams = false;
  if (noNormalize)
    _normalize = false;
  if (noState)
    _includeStateNgrams = false;
  
  return 0;
}

SparseRealVec* SentenceAlignmentFeatureGen::getFeatures(const Pattern& x,
    Label y, int iPrev, int jPrev, int iNew, int jNew, const EditOperation& op,
    const vector<AlignmentPart>& history) {
    
  // TODO: May want to do the tokenizing in the reader, store in a
  // different data structure (not StringPair) possibly with
  // feature ids for the "basic" features, which can then be
  // combined into "complex" features here in the FeatureGen.
  // Note that this would require making AlignmentTransducer more
  // generic, since it currently assumes two vectors of strings;
  // however, I think it only actually needs to know the "size" of
  // the source and target sequences.
    
  const int histLen = history.size();
  assert(iNew >= 0 && jNew >= 0);
  assert(histLen >= 1);
  assert(_order >= 0);
  
  set<int> featureIds;

  // TODO: See if FastFormat (or some other int2str method) improves efficiency
  // http://stackoverflow.com/questions/191757/c-concatenate-string-and-int
  
  // Determine the point in the history where the longest n-gram begins.
  int left;
  if (_order + 1 > histLen)
    left = 0;
  else
    left = histLen - (_order + 1);
  assert(left >= 0);
  
  // Extract n-grams of the state sequence (up to the Markov order).
  if (_includeStateNgrams) {
    string s;
    for (int k = histLen - 1; k >= left; k--) {
      s = history[k].opName + s;
      addFeatureId("S:" + s, y, featureIds);
      s = FeatureGenConstants::OP_SEP + s;
    }
  }

  // These features only fire if we didn't perform a no-op on this arc.
  if (op.getId() != OpNone::ID) {
    if (_includeAlignNgrams) {
      typedef tokenizer<char_separator<char> > Tokenizer;
      char_separator<char> featSep(FeatureGenConstants::WORDFEAT_SEP);
      char_separator<char> phraseSep(FeatureGenConstants::PHRASE_SEP);
      
      // Figure out how many indicator features there are. We assume that the
      // first feature name is "WRD" -- the id of the word itself. Since this
      // is a very sparse feature, we choose to ignore it below. However, it
      // could be used in the future to, e.g., lookup the word in some sort of
      // table to get a quantity that's of interest.
      int numFeats = 0;
      const AlignmentPart& edit = history.back();
      bool gotSource = edit.source != FeatureGenConstants::EPSILON;
      bool gotTarget = edit.target != FeatureGenConstants::EPSILON;
      // We assume the source and target have the same number of features, so
      // we only need to count the features in one or the other.
      if (gotSource) {
        Tokenizer phrases(edit.source, phraseSep);
        const string phrase0 = *phrases.begin();
        Tokenizer tokens(phrase0, featSep);
        Tokenizer::const_iterator it = tokens.begin();
        assert(istarts_with(*it, "WRD")); // FIXME: Should not be hard-coded
        for (; it != tokens.end(); ++it)
          ++numFeats;
      }
      else {
        assert(gotTarget);
        Tokenizer phrases(edit.target, phraseSep);
        const string phrase0 = *phrases.begin();
        Tokenizer tokens(phrase0, featSep);
        Tokenizer::const_iterator it = tokens.begin();
        assert(istarts_with(*it, "WRD")); // FIXME: Should not be hard-coded
        for (; it != tokens.end(); ++it)
          ++numFeats;
      }      
      --numFeats; // ignore WRD
      assert(numFeats > 0);
      
      vector<string> sourceFeats(numFeats);
      vector<string> targetFeats(numFeats);
      vector<string> ngramFeats(numFeats);
      
      for (int k = histLen - 1, n = 1; k >= left; k--, n++) {
        gotSource = history[k].source != FeatureGenConstants::EPSILON;
        gotTarget = history[k].target != FeatureGenConstants::EPSILON;
          
        if (gotSource) {
          Tokenizer phrases(history[k].source, phraseSep);
          BOOST_FOREACH(string phrase, phrases) {
            Tokenizer tokens(phrase, featSep);
            Tokenizer::const_iterator it = tokens.begin();
            ++it; // skip WRD
            for (int fi = 0; fi < numFeats; fi++)
              sourceFeats[fi] += (*it++) + FeatureGenConstants::PHRASE_SEP;
            assert(it == tokens.end());
          }
        }
        else {
          for (int fi = 0; fi < numFeats; fi++)
            sourceFeats[fi] = FeatureGenConstants::EPSILON;
        }
        
        if (gotTarget) {
          Tokenizer phrases(history[k].target, phraseSep);
          BOOST_FOREACH(string phrase, phrases) {
            Tokenizer tokens(phrase, featSep);
            Tokenizer::const_iterator it = tokens.begin();
            ++it; // skip WRD
            for (int fi = 0; fi < numFeats; fi++)
              targetFeats[fi] += (*it++) + FeatureGenConstants::PHRASE_SEP;
            assert(it == tokens.end());
          }
        }
        else {
          for (int fi = 0; fi < numFeats; fi++)
            targetFeats[fi] = FeatureGenConstants::EPSILON;
        }
        
        for (int fi = 0; fi < numFeats; fi++) {
          ngramFeats[fi] = sourceFeats[fi] + FeatureGenConstants::OP_SEP +
              targetFeats[fi] + ngramFeats[fi]; 
          // The last condition in the if statements implies that we don't fire an
          // alignment unigram feature if the operation was a noop.
          if (_includeAlignNgrams && (!_alignUnigramsOnly || n == 1) &&
              (op.getId() != OpNone::ID || n > 1)) {
            addFeatureId("A:" + ngramFeats[fi], y, featureIds);
          }
          ngramFeats[fi] = FeatureGenConstants::PART_SEP + ngramFeats[fi];
        }
      }
    }
  }
  
  const size_t d = _alphabet->size();
  SparseRealVec* fv = new SparseRealVec(d);
  set<int>::const_iterator it;
  for (it = featureIds.begin(); it != featureIds.end(); ++it) {
    const size_t index = *it;
    if (index >= d) {
      // Assume we're in feature gathering mode, in which case the fv we return
      // will be ignored.
      return fv;
    }
    (*fv)(index) = 1.0;
  }

  if (_normalize) {
    const double normalization = x.getSize();
    assert(normalization > 0);
    (*fv) /= normalization;
  }
  
  return fv;
}

inline double SentenceAlignmentFeatureGen::getDefaultFeatureWeight(
    const string& f, Label label) const {
  return 0.0;
}

inline void SentenceAlignmentFeatureGen::addFeatureId(const string& f, Label y,
    set<int>& featureIds) const {
  const int fId = _alphabet->lookup(f, y, true);
  if (fId == -1)
    return;
  featureIds.insert(fId);
}
