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


#include <string>
using namespace std;

class StateType {
  public:
    StateType(int id, string name) : id(id), name(name) { }
    
    int getId() const { return id; }
    
    const string& getName() const { return name; }


  private:
    int id;
    
    string name;

};

#endif
