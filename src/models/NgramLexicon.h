/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _NGRAMLEXICON_H
#define _NGRAMLEXICON_H

#include <boost/unordered_set.hpp>
#include <string>

class NgramLexicon {

  public:
    NgramLexicon(std::string filename);
    
    bool contains(const std::string& ngram) const;
    
    static std::string getNgramString(const std::vector<std::string>& ngram,
        int len, int start = 0);
    
  private:
    boost::unordered_set<std::string> _ngrams;
};

#endif
