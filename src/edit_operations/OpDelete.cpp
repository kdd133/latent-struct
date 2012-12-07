/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "EditOperation.h"
#include "OpDelete.h"
#include <assert.h>
#include <boost/regex.hpp>
#include <string>
#include <vector>

using namespace std;

OpDelete::OpDelete(int opId, const StateType* defaultDestinationState,
    string name, int phraseLengthSource) :
    EditOperation(opId, name, defaultDestinationState),
    _phraseLengthSource(phraseLengthSource),
    _conditionEnabled(false),
    _acceptMatching(false) {
}

void OpDelete::setCondition(string tokenRegexStr, bool acceptMatching) {
  if (tokenRegexStr.size() > 0) {
    _conditionEnabled = true;
    _tokenRegex = boost::regex(tokenRegexStr);
    _acceptMatching = acceptMatching;
  }
}

const StateType* OpDelete::apply(const vector<string>& source,
    const vector<string>& target, const StateType* prevStateType,
    const int i, const int j, int& iNew, int& jNew) const {
  if (i + _phraseLengthSource > source.size())
    return 0;
  if (_conditionEnabled) {
    for (int l = 0; l < _phraseLengthSource; l++) {
      if (boost::regex_match(source[i + l], _tokenRegex)) {
        if (!_acceptMatching)
          return 0;
      }
      else if (_acceptMatching)
        return 0;
    }
  }
  iNew = i + _phraseLengthSource;
  jNew = j;
  return _defaultDestinationState;
}
