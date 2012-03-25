/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPNONE_H
#define _OPNONE_H

#include "EditOperation.h"
#include <assert.h>
#include <string>
#include <vector>
using namespace std;

class OpNone : public EditOperation {

  public:
  
    OpNone(string name = "None") : EditOperation(ID, name) {}
    
    virtual const StateType* apply(const vector<string>& source,
                                   const vector<string>& target,
                                   const StateType* prevStateType,
                                   const int i,
                                   const int j,
                                   int& iNew,
                                   int& jNew) const {
      iNew = i;
      jNew = j;
      assert(prevStateType != 0);
      return prevStateType;
    }
    
    static const int ID;              
};

#endif
