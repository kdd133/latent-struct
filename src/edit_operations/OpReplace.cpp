/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "EditOperation.h"
#include "OpReplace.h"
#include <boost/regex.hpp>
#include <string>
#include <vector>

using namespace std;

OpReplace::OpReplace(int opId, const StateType* defaultDestinationState,
    string name, int phraseLengthSource, int phraseLengthTarget) :
    EditOperation(opId, name, defaultDestinationState),
    _phraseLengthSource(phraseLengthSource),
    _phraseLengthTarget(phraseLengthTarget),
    _conditionEnabledSource(false),
    _conditionEnabledTarget(false),
    _acceptMatchingSource(false),
    _acceptMatchingTarget(false) {
}

void OpReplace::setCondition(string tokenRegexStrSource,
    string tokenRegexStrTarget, bool acceptMatchingSource,
    bool acceptMatchingTarget) {
  if (tokenRegexStrSource.size() > 0) {
    _conditionEnabledSource = true;
    _tokenRegexSource = boost::regex(tokenRegexStrSource);
    _acceptMatchingSource = acceptMatchingSource;
  }
  if (tokenRegexStrTarget.size() > 0) {
    _conditionEnabledTarget = true;
    _tokenRegexTarget = boost::regex(tokenRegexStrTarget);
    _acceptMatchingTarget = acceptMatchingTarget;
  }
}

const StateType* OpReplace::apply(const vector<string>& source,
    const vector<string>& target, const StateType* prevStateType,
    const int i, const int j, int& iNew, int& jNew) const {
  if (i + _phraseLengthSource > source.size() ||
      j + _phraseLengthTarget > target.size())
    return 0;
  if (_conditionEnabledSource || _conditionEnabledTarget) {
    if (_conditionEnabledSource) {
      for (int l = 0; l < _phraseLengthSource; l++) {
        if (boost::regex_match(source[i + l], _tokenRegexSource)) {
          if (!_acceptMatchingSource)
            return 0;
        }
        else if (_acceptMatchingSource)
          return 0;
      }
    }
    if (_conditionEnabledTarget) {
      for (int l = 0; l < _phraseLengthTarget; l++) {
        if (boost::regex_match(target[j + l], _tokenRegexTarget)) {
          if (!_acceptMatchingTarget)
            return 0;
        }
        else if (_acceptMatchingTarget)
          return 0;
      }
    }
  }
  // If an n-gram lexicon is present, we only apply the operation if the phrase
  // is contained in the lexicon. Note: All unigrams are allowed by default.
  else if (_nglexSource || _nglexTarget) {
    if (_nglexSource && _phraseLengthSource > 1) {
      string ngram = NgramLexicon::getNgramString(source, _phraseLengthSource, i);
      if (!_nglexSource->contains(ngram))
        return 0;
    }
    if (_nglexTarget && _phraseLengthTarget > 1) {
      string ngram = NgramLexicon::getNgramString(target, _phraseLengthTarget, j);
      if (!_nglexTarget->contains(ngram))
        return 0;
    }
  }
  iNew = i + _phraseLengthSource;
  jNew = j + _phraseLengthTarget;
  return _defaultDestinationState;
}
