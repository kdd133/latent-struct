/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _STRINGPAIRPHRASES_H
#define _STRINGPAIRPHRASES_H

#include "FeatureGenConstants.h"
#include "StringPairAligned.h"
#include "Utility.h"
#include <ostream>
#include <string>
#include <vector>

// A data structure that stores an aligned pair of strings, which have been
// augmented with "-" symbols that indicate insertions and deletions.
class StringPairPhrases : public StringPairAligned {
  public:
    StringPairPhrases(std::vector<std::string> source,
        std::vector<std::string> target, std::vector<std::string> phrasePairs) :
        StringPairAligned(source, target), _phrasePairs(phrasePairs) {}
    
    virtual const std::vector<std::string>& getPhrasePairs() const;
    
  private:
    
    std::vector<std::string> _phrasePairs;
};

inline const std::vector<std::string>& StringPairPhrases::getPhrasePairs() const {
  return _phrasePairs;
}

#endif
