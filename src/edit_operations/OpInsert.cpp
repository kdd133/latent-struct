/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "EditOperation.h"
#include "OpInsert.h"
#include <boost/regex.hpp>
#include <string>
#include <vector>

using namespace std;

OpInsert::OpInsert(int opId, const StateType* defaultDestinationState,
    string name, int phraseLengthTarget) :
    EditOperation(opId, name, defaultDestinationState),
    _phraseLengthTarget(phraseLengthTarget),
    _conditionEnabled(false),
    _acceptMatching(false) {
}

void OpInsert::setCondition(string tokenRegexStr, bool acceptMatching) {
  if (tokenRegexStr.size() > 0) {
    _conditionEnabled = true;
    _tokenRegex = boost::regex(tokenRegexStr);
    _acceptMatching = acceptMatching;
  }
}

const StateType* OpInsert::apply(const vector<string>& source,
    const vector<string>& target, const StateType* prevStateType,
    const int i, const int j, int& iNew, int& jNew) const {
  if (j + _phraseLengthTarget > target.size())
    return 0;
  if (_conditionEnabled) {
    for (int l = 0; l < _phraseLengthTarget; l++) {
      if (boost::regex_match(target[j + l], _tokenRegex)) {
        if (!_acceptMatching)
          return 0;
      }
      else if (_acceptMatching)
        return 0;
    }
  }
  // If an n-gram lexicon is present, we only apply the operation if the phrase
  // is contained in the lexicon. Note: All unigrams are allowed by default.
  else if (_nglexTarget && _phraseLengthTarget > 1) {
    string ngram = NgramLexicon::getNgramString(target, _phraseLengthTarget, j);
    if (!_nglexTarget->contains(ngram))
      return 0;
  }
  iNew = i;
  jNew = j + _phraseLengthTarget;
  return _defaultDestinationState;
}
