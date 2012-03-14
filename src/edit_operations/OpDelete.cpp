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
#include <list>
#include <string>
#include <vector>
using namespace std;

OpDelete::OpDelete(int opId, int defaultDestinationStateId, string name,
    int phraseLengthSource, list<int> cantFollowStateTypeIds) :
    EditOperation(opId, name),
    _defaultDestinationStateId(defaultDestinationStateId),
    _phraseLengthSource(phraseLengthSource),
    _cantFollowStateTypeIds(cantFollowStateTypeIds),
    _conditionEnabled(false) {
}

void OpDelete::setCondition(string tokenRegexStr, bool acceptMatching) {
  if (tokenRegexStr.size() > 0) {
    _conditionEnabled = true;
    _tokenRegex = boost::regex(tokenRegexStr);
    _acceptMatching = acceptMatching;
  }
}

int OpDelete::apply(const vector<string>& source, const vector<string>& target,
    const int prevStateTypeId, const int i, const int j, int& iNew, int& jNew) const {
  if (_cantFollowStateTypeIds.size() > 0) {
    list<int>::const_iterator it = _cantFollowStateTypeIds.begin();
    for (; it != _cantFollowStateTypeIds.end(); ++it) {
      assert(*it != -1);
      if (*it == prevStateTypeId)
        return -1;
    }
  }
  if (i + _phraseLengthSource > source.size())
    return -1;
  if (_conditionEnabled) {
    for (int l = 0; l < _phraseLengthSource; l++) {
      if (boost::regex_match(source[i + l], _tokenRegex)) {
        if (!_acceptMatching)
          return -1;
      }
      else if (_acceptMatching)
        return -1;
    }
  }
  iNew = i + _phraseLengthSource;
  jNew = j;
  return _defaultDestinationStateId;
}
