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
#include <string>
#include <vector>
using namespace std;

OpInsert::OpInsert(int opId, int defaultDestinationStateId, string name,
    int phraseLengthTarget, int cantFollowStateTypeId) :
    EditOperation(opId, name),
    _defaultDestinationStateId(defaultDestinationStateId),
    _phraseLengthTarget(phraseLengthTarget),
    _cantFollowStateTypeId(cantFollowStateTypeId) {
}

int OpInsert::apply(const vector<string>& source, const vector<string>& target,
    const int prevStateTypeId, const int i, const int j, int& iNew, int& jNew) const {
  if (_cantFollowStateTypeId != -1 && _cantFollowStateTypeId == prevStateTypeId)
    return -1;
  const int T = target.size();
  if (j + _phraseLengthTarget > T)
    return -1;
  iNew = i;
  jNew = j + _phraseLengthTarget;
  return _defaultDestinationStateId;
}
