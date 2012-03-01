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
#include <string>
#include <vector>
using namespace std;

OpDelete::OpDelete(int opId, int defaultDestinationStateId, string name,
    int phraseLengthSource, int cantFollowStateTypeId) :
    EditOperation(opId, name),
    _defaultDestinationStateId(defaultDestinationStateId),
    _phraseLengthSource(phraseLengthSource),
    _cantFollowStateTypeId(cantFollowStateTypeId) {
}

int OpDelete::apply(const vector<string>& source, const vector<string>& target,
    const int prevStateTypeId, const int i, const int j, int& iNew, int& jNew) const {
  if (_cantFollowStateTypeId != -1 && _cantFollowStateTypeId == prevStateTypeId)
    return -1;
  const int S = source.size();  
  if (i + _phraseLengthSource > S)
    return -1;
  iNew = i + _phraseLengthSource;
  jNew = j;
  return _defaultDestinationStateId;
}
