/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPIDENTITY_H
#define _OPIDENTITY_H

#include "EditOperation.h"
#include <string>
#include <vector>
using namespace std;

// An operation that is applied if the source and target phrases are identical.
class OpMatch : public EditOperation {

  public:
  
    OpMatch(int opId, int defaultDestinationStateId, string name =
      "Match", int phraseLengthSource = 1, int phraseLengthTarget = 1,
      int cantFollowStateTypeId = -1);
    
    virtual ~OpMatch() {}
    
    int apply(const vector<string>& source,
              const vector<string>& target,
              const int prevStateTypeId,
              const int i,
              const int j,
              int& iNew,
              int& jNew) const;
              
  private:
  
    int _defaultDestinationStateId;
    
    int _phraseLengthSource;
    
    int _phraseLengthTarget;
    
    int _cantFollowStateTypeId;
};

#endif
