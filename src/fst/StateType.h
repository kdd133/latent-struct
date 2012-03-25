/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _STATE_H
#define _STATE_H

#include "EditOperation.h"
#include <list>
#include <string>
using std::list;
using std::string;

class StateType {

  public:
  
    StateType(string name) : _id(-1), _name(name) {}
  
    StateType(int id, string name) : _id(id), _name(name) {}
    
    int getId() const {
      return _id;
    }
    
    void setId(int id) {
      _id = id;
    }
    
    const string& getName() const {
      return _name;
    }
    
    void addValidOperation(const EditOperation* op) {
      _validOps.push_back(op);
    }
    
    const list<const EditOperation*>& getValidOperations() const {
      return _validOps;
    }


  private:
  
    int _id;
    
    string _name;
    
    // Each pointer in this list represents a valid/legal edit operation that
    // may be applied while in this state. We do not own the EditOperation
    // objects and are therefore not responsible for deleting them.
    list<const EditOperation*> _validOps;
    
    // We disallow copying in order to avoid accidental (and unnecessary)
    // expense of copying the list member.
    StateType& operator=(const StateType& other);
    StateType(const StateType& other);
};

#endif
