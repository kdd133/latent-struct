/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "EditOperation.h"
#include "OpMatch.h"
#include <boost/regex.hpp>
#include <string>
#include <vector>
using namespace std;

OpMatch::OpMatch(int opId, const StateType* defaultDestinationState,
    string name, int phraseLength) :
    EditOperation(opId, name, defaultDestinationState),
    _phraseLength(phraseLength),
    _conditionEnabled(false) {
}

void OpMatch::setCondition(string tokenRegexStr, bool acceptMatching) {
  if (tokenRegexStr.size() > 0) {
    _conditionEnabled = true;
    _tokenRegex = boost::regex(tokenRegexStr);
    _acceptMatching = acceptMatching;
  }
}

const StateType* OpMatch::apply(const vector<string>& source,
    const vector<string>& target, const StateType* prevStateType,
    const int i, const int j, int& iNew, int& jNew) const {
  if (i + _phraseLength > source.size() || j + _phraseLength > target.size())
    return 0;
  // If the two phrases are not identical, return -1.
  for (int l = 0; l < _phraseLength; l++) {
    if (source[i + l] != target[j + l])
      return 0;
  }
  if (_conditionEnabled) {
    // Since we now know that the source and target phrases are equal, we can
    // just check the matching condition on the source side.
    for (int l = 0; l < _phraseLength; l++) {
      if (boost::regex_match(source[i + l], _tokenRegex)) {
        if (!_acceptMatching)
          return 0;
      }
      else if (_acceptMatching)
        return 0;
    }
  }
  iNew = i + _phraseLength;
  jNew = j + _phraseLength;
  return _defaultDestinationState;
}
