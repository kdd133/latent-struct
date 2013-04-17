/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "LogWeight.h"
#include "Utility.h"
#include <assert.h>
#include <cmath>
#include <limits>
#include <ostream>

LogWeight::LogWeight() : _val(-std::numeric_limits<double>::infinity()) {
}

LogWeight::LogWeight(double value, bool valueIsLog) {
  if (valueIsLog)
    _val = value;
  else {
    // 0 should not be passed as an argument when valueIsLog=false, since the
    // default constructor gives the same result without calling log().
    assert(value > 0);
    _val = log(value);
  }
}

// See Table 3 in Li & Eisner paper titled:
// "First- and Second-Order Expectation Semirings with Applications..."
const LogWeight LogWeight::operator+(const LogWeight& w) const {
  const LogWeight zero;
  if ((*this) == zero)
    return w;
  else if (w == zero)
    return (*this);
  else {
    double la; // natural log of a
    double lb; // natural log of b
    // Choose a and b s.t. log(a) >= log(b).
    if (_val >= w._val) {
      la = _val;
      lb = w._val;
    }
    else {
      la = w._val;
      lb = _val;
    }
    const double negDiff = lb - la;
    if (negDiff < -20) {
      // If b is much smaller than a, then ignore b.
      // See https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
      return LogWeight(la, true);
    }
    return LogWeight(la + Utility::log1Plus(exp(negDiff)), true);
  }
}

const LogWeight LogWeight::operator*(const LogWeight& w) const {
  // The explicit check for zero isn't needed because the addition below will
  // return -Inf (i.e., "zero") if either of the operands is -Inf.
  // const LogWeight zero(0);
  // if ((*this) == zero || w == zero)
  //   return zero;
  return LogWeight(_val + w._val, true);
}

const LogWeight LogWeight::operator/(const LogWeight& w) const {
  // TODO: Handle the case where the argument is zero. Should division by zero
  // return NaN or throw an exception?
  return LogWeight(_val - w._val, true);
}

LogWeight& LogWeight::operator+=(const LogWeight& w) {
  const LogWeight result = (*this) + w;
  _val = result._val;
  return (*this);
}

LogWeight& LogWeight::operator*=(const LogWeight& w) {
  const LogWeight result = (*this) * w;
  _val = result._val;
  return (*this);
}

LogWeight& LogWeight::operator/=(const LogWeight& w) {
  const LogWeight result = (*this) / w;
  _val = result._val;
  return (*this);
}

std::ostream& operator<<(std::ostream& out, const LogWeight& w) {
  return out << w._val;
}

LogWeight sqrt(LogWeight w) {
  const double root = sqrt(exp((double)w));
  return LogWeight(root);
}

LogWeight abs(LogWeight w) {
  // abs(w) = log(abs(exp(w))) = log(exp(w)) = w
  //   Where the second equality follows from the fact that exp(w) is positive
  //   for any w.
  return w;
}
