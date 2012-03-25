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
#include <boost/ptr_container/ptr_list.hpp>
#include <string>
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
    
    void addValidOperation(EditOperation* op) {
      _validOps.push_back(op);
    }
    
    const boost::ptr_list<EditOperation>& getValidOperations() const {
      return _validOps;
    }


  private:
  
    int _id;
    
    string _name;
    
    boost::ptr_list<EditOperation> _validOps;
    
    // Copying is complicated by the ptr_list, so we'll just disallow it.
    StateType& operator=(const StateType& other);
    StateType(const StateType& other);
};

#endif
