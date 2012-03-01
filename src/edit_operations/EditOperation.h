/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _EDITOPERATION_H
#define _EDITOPERATION_H


#include <string>
#include <vector>
using namespace std;

class StateType;
class StringPair;

class EditOperation {

  public:
  
    EditOperation(int id, string name) : _id(id), _name(name) {}
    
    virtual ~EditOperation() {}
    
    // Returns a non-negative state Id if the operation could be legally
    // applied; or, -1 otherwise.
    //
    //i: Position in source string.
    //j: Position in target string.
    //iNew: Position in source string after applying the operation.
    //jNew: Position in target string after applying the operation.
    virtual int apply(const vector<string>& source,
                      const vector<string>& target,
                      const int prevStateTypeId,
                      const int i,
                      const int j,
                      int& iNew,
                      int& jNew) const = 0;

    int getId() const { return _id; }
    
    //Returns the name that uniquely identifies this edit operation.
    const string& getName() const { return _name; }

    static int noopId() {
      static const int NOOP = -1;
      return NOOP;
    }

  protected:
  
    int _id;
    
    string _name;

};
#endif
