/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _STRINGPAIRALIGNED_H
#define _STRINGPAIRALIGNED_H

#include "Pattern.h"
#include "Utility.h"
#include <ostream>
#include <string>
#include <vector>

// A data structure that stores an aligned pair of strings, which have been
// augmented with "-" symbols that indicate insertions and deletions.
class StringPairAligned : public Pattern {
  public:
    StringPairAligned(std::vector<std::string> source,
        std::vector<std::string> target) {
      Utility::levenshtein(source, target, _source, _target, _substitutionCost);
    }
    
    const std::vector<std::string>& getSource() const;

    const std::vector<std::string>& getTarget() const;
    
    // Returns the length of the longer string.
    virtual int getSize() const;
    
    friend std::ostream& operator<<(std::ostream& out,
        const StringPairAligned& s);


  private:
    std::vector<std::string> _source;

    std::vector<std::string> _target;
    
    static const int _substitutionCost = 99999;

};

inline const std::vector<std::string>& StringPairAligned::getSource() const {
  return _source;
}

inline const std::vector<std::string>& StringPairAligned::getTarget() const {
  return _target;
}

inline int StringPairAligned::getSize() const {
  return _source.size() > _target.size() ? _source.size() : _target.size();
}

inline std::ostream& operator<<(std::ostream& out, const StringPairAligned& s) {
  for (std::size_t i = 0; i < s.getSize(); ++i)
    out << s._source[i] << " ";
  out << std::endl;
  for (std::size_t i = 0; i < s.getSize(); ++i)
    out << s._target[i] << " ";
  out << std::endl;
  return out;
}

#endif
