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

class EditOperation {

  public:
  
    EditOperation(int id, string name, const StateType* defaultDestState = 0)
      : _id(id), _name(name), _defaultDestinationState(defaultDestState) {}
    
    virtual ~EditOperation() {}
    
    virtual const StateType* apply(const vector<string>& source,
                                   const vector<string>& target,
                                   const StateType* prevStateType,
                                   const int i,
                                   const int j,
                                   int& iNew,
                                   int& jNew) const = 0;

    int getId() const {
      return _id;
    }
    
    //Returns the name that uniquely identifies this edit operation.
    const string& getName() const {
      return _name;
    }
    
    const StateType* getDefaultDestinationState() const {
      return _defaultDestinationState;
    }

  protected:
  
    int _id;
    
    string _name;
    
    const StateType* _defaultDestinationState;
};
#endif
