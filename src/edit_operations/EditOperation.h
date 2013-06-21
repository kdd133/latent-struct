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


#include "NgramLexicon.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

class StateType;

class EditOperation {

  public:
  
    EditOperation(int id, std::string name,
      const StateType* defaultDestState = 0) : _id(id), _name(name),
      _defaultDestinationState(defaultDestState) {}
    
    virtual ~EditOperation() {}
    
    virtual const StateType* apply(const std::vector<std::string>& source,
                                   const std::vector<std::string>& target,
                                   const StateType* prevStateType,
                                   const int i,
                                   const int j,
                                   int& iNew,
                                   int& jNew) const = 0;

    int getId() const {
      return _id;
    }
    
    //Returns the name that uniquely identifies this edit operation.
    const std::string& getName() const {
      return _name;
    }
    
    const StateType* getDefaultDestinationState() const {
      return _defaultDestinationState;
    }
    
    void setNgramLexicons(boost::shared_ptr<NgramLexicon> lexiconSource,
        boost::shared_ptr<NgramLexicon> lexiconTarget) {
      _nglexSource = lexiconSource;
      _nglexTarget = lexiconTarget;
    }
    
  protected:
  
    int _id;
    
    std::string _name;
    
    const StateType* _defaultDestinationState;
    
    boost::shared_ptr<NgramLexicon> _nglexSource;
    
    boost::shared_ptr<NgramLexicon> _nglexTarget;
};
#endif
