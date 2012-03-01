/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPDELETE_H
#define _OPDELETE_H

#include "EditOperation.h"
#include <string>
#include <vector>
using namespace std;

class OpDelete : public EditOperation {
  public:
    OpDelete(int opId, int defaultDestinationStateId, string name = "Delete",
        int phraseLengthSource = 1, int cantFollowStateTypeId = -1);
    
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
    
    int _cantFollowStateTypeId;
};

#endif
