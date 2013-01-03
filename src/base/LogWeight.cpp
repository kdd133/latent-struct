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
#include <ostream>

LogWeight::LogWeight(double value, bool valueIsLog) {
  if (valueIsLog)
    _val = value;
  else {
    assert(value >= 0);
    _val = log(value);
  }
}

// See Table 3 in Li & Eisner paper titled:
// "First- and Second-Order Expectation Semirings with Applications..."
const LogWeight LogWeight::operator+(const LogWeight& w) const {
  const LogWeight zero(0);
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
    const double x = exp(lb - la);
    
    LogWeight result;
    result._val = la + Utility::log1Plus(x);
    return result;
  }
}

const LogWeight LogWeight::operator*(const LogWeight& w) const {
  // The explicit check for zero isn't needed because the addition below will
  // return -Inf (i.e., "zero") if either of the operands is -Inf.
  // const LogWeight zero(0);
  // if ((*this) == zero || w == zero)
  //   return zero;
  LogWeight result;
  result._val = _val + w._val;
  return result;
}

const LogWeight LogWeight::operator/(const LogWeight& w) const {
  // TODO: Handle the case where the argument is zero. Should division by zero
  // return NaN or throw an exception?
  LogWeight result;
  result._val = _val - w._val;
  return result;
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
  double root = sqrt(exp((double)w));
  return LogWeight(root);
}

LogWeight abs(LogWeight w) {
  // abs(w) = log(abs(exp(w))) = log(exp(w)) = w
  //   Where the second equality follows from the fact that exp(w) is positive
  //   for any w.
  return w;
}
