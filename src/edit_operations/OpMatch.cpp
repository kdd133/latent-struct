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
#include <string>
#include <vector>
using namespace std;

OpMatch::OpMatch(int opId, int defaultDestinationStateId, string name,
    int phraseLengthSource, int phraseLengthTarget, int cantFollowStateTypeId) :
    EditOperation(opId, name),
    _defaultDestinationStateId(defaultDestinationStateId),
    _phraseLengthSource(phraseLengthSource),
    _phraseLengthTarget(phraseLengthTarget),
    _cantFollowStateTypeId(cantFollowStateTypeId) {
}

int OpMatch::apply(const vector<string>& source, const vector<string>& target,
    const int prevStateTypeId, const int i, const int j, int& iNew, int& jNew) const {
  if (_cantFollowStateTypeId != -1 && _cantFollowStateTypeId == prevStateTypeId)
    return -1;
  const int S = source.size();
  const int T = target.size();  
  if (i + _phraseLengthSource > S || j + _phraseLengthTarget > T)
    return -1;
  // if the phrase lengths differ, this can't be an identical match
  if (_phraseLengthSource != _phraseLengthTarget)
    return -1;    
  int posSource = i;
  int posTarget = j;
  int l = 0;
  bool same = true;
  for (; l < _phraseLengthSource; l++) {
    if (source[posSource+l] != target[posTarget+l]) {
      same = false;
      break;
    }
  }
  if (!same)
    return -1;
  iNew = i + _phraseLengthSource;
  jNew = j + _phraseLengthTarget;
  return _defaultDestinationStateId;
}
