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

OpReplace::OpReplace(int opId, int defaultDestinationStateId, string name,
    int phraseLengthSource, int phraseLengthTarget) :
    EditOperation(opId, name),
    _defaultDestinationStateId(defaultDestinationStateId),
    _phraseLengthSource(phraseLengthSource),
    _phraseLengthTarget(phraseLengthTarget),
    _conditionEnabledSource(false),
    _conditionEnabledTarget(false) {
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

int OpReplace::apply(const vector<string>& source, const vector<string>& target,
    const int prevStateTypeId, const int i, const int j, int& iNew, int& jNew) const {
  if (i + _phraseLengthSource > source.size() ||
      j + _phraseLengthTarget > target.size())
    return -1;
  if (_conditionEnabledSource) {
    for (int l = 0; l < _phraseLengthSource; l++) {
      if (boost::regex_match(source[i + l], _tokenRegexSource)) {
        if (!_acceptMatchingSource)
          return -1;
      }
      else if (_acceptMatchingSource)
        return -1;
    }
  }
  if (_conditionEnabledTarget) {
    for (int l = 0; l < _phraseLengthTarget; l++) {
      if (boost::regex_match(target[j + l], _tokenRegexTarget)) {
        if (!_acceptMatchingTarget)
          return -1;
      }
      else if (_acceptMatchingTarget)
        return -1;
    }
  }
  iNew = i + _phraseLengthSource;
  jNew = j + _phraseLengthTarget;
  return _defaultDestinationStateId;
}
