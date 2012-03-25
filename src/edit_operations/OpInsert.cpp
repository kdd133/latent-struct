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

OpInsert::OpInsert(int opId, int defaultDestinationStateId, string name,
    int phraseLengthTarget) :
    EditOperation(opId, name, defaultDestinationStateId),
    _phraseLengthTarget(phraseLengthTarget),
    _conditionEnabled(false) {
}

void OpInsert::setCondition(string tokenRegexStr, bool acceptMatching) {
  if (tokenRegexStr.size() > 0) {
    _conditionEnabled = true;
    _tokenRegex = boost::regex(tokenRegexStr);
    _acceptMatching = acceptMatching;
  }
}

int OpInsert::apply(const vector<string>& source, const vector<string>& target,
    const int prevStateTypeId, const int i, const int j, int& iNew, int& jNew) const {
  if (j + _phraseLengthTarget > target.size())
    return -1;
  if (_conditionEnabled) {
    for (int l = 0; l < _phraseLengthTarget; l++) {
      if (boost::regex_match(target[j + l], _tokenRegex)) {
        if (!_acceptMatching)
          return -1;
      }
      else if (_acceptMatching)
        return -1;
    }
  }
  iNew = i;
  jNew = j + _phraseLengthTarget;
  return _defaultDestinationStateId;
}
