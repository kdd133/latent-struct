/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPMATCH_H
#define _OPMATCH_H

#include "EditOperation.h"
#include <string>
#include <vector>
using namespace std;

// Replace is an operation that generalizes Substitute and Match. That is,
// Replace is applied regardless of whether the source phrase and the target
// phrase are identical or distinct.
class OpReplace : public EditOperation {

  public:
  
    OpReplace(int opId, int defaultDestinationStateId, string name = "Replace",
      int phraseLengthSource = 1, int phraseLengthTarget = 1,
      int cantFollowStateTypeId = -1);
    
    virtual ~OpReplace() {}
    
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
