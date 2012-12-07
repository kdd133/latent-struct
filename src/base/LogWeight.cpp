/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "LogWeight.h"
#include "RealWeight.h"
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

RealWeight LogWeight::convert() const {
  return RealWeight(exp(_val));
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

const LogWeight LogWeight::operator-(const LogWeight& w) const {
  return (*this) + (-w);
}

const LogWeight LogWeight::operator*(const LogWeight& w) const {
  const LogWeight zero(0);
  if ((*this) == zero || w == zero)
    return zero;
  else {
    LogWeight result;
    result._val = _val + w._val;
    return result;
  }
}

const LogWeight LogWeight::operator/(const LogWeight& w) const {
  return (*this) * (-w);
}

LogWeight& LogWeight::operator+=(const LogWeight& w) {
  const LogWeight result = (*this) + w;
  _val = result._val;
  return (*this);
}

LogWeight& LogWeight::operator-=(const LogWeight& w) {
  const LogWeight result = (*this) - w;
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

LogWeight operator-(const LogWeight& w) {
  LogWeight negated = w;
  negated._val *= -1;
  return negated;
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
