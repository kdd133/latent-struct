/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "BmrmOptimizer.h"
#include "FeatureVector.h"
#include "Model.h"
#include "RealWeight.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/program_options.hpp>
#include <boost/ptr_container/ptr_deque.hpp>
#include <boost/timer/timer.hpp>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <string>
#include <uQuadProg++.hh>
using namespace std;

BmrmOptimizer::BmrmOptimizer(TrainingObjective& objective) :
    Optimizer(objective, 1e-4), _maxIters(250) {
}

int BmrmOptimizer::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  options.add_options()
    ("max-iters", opt::value<size_t>(&_maxIters)->default_value(250),
        "maximum number of iterations")
    ("no-shrinking", opt::bool_switch(&_noShrinking),
        "disable the shrinking heuristic")
    ("quiet", opt::bool_switch(&_quiet), "suppress optimizer output")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help"))
    cout << options << endl;
  return 0;
}

double BmrmOptimizer::train(WeightVector& w, double tol) const {
  namespace ublas = boost::numeric::ublas;
  const size_t d = w.getDim();
  assert(d > 0);

  // Some variables we'll need to reuse inside the main loop.
  boost::ptr_deque<ublas::vector<double> > grads;
  ublas::vector<double> b(1);
  ublas::matrix<double> G(1, 1);
  ublas::matrix<double> copyG(1, 1);
  ublas::vector<double> wTemp(d);
  FeatureVector<RealWeight> grad_t(d);
  double Remp;
  
  // Compute the initial objective value and gradient.
  _objective.valueAndGradient(w, Remp, grad_t);
  double min_Jw = (0.5 * _beta * w.squaredL2Norm()) + Remp;
  double Jw = 0; // initialized in the loop
  
  if (!_quiet)
    cout << name() << ": Starting objective value is " << min_Jw << endl;
  
  const double tiny = 1e-12; // Add this value to D to ensure pos-def.
  bool converged = false;
  bool dropped = false; // Did we discard any gradient vectors during prev t?
  for (size_t t = 1; t <= _maxIters; t++) {
    boost::timer::auto_cpu_timer timer;
    
    // Add the next column to the matrix "A", which we'll actually represent as
    // a ptr_vector of column vectors.
    ublas::vector<double>* g_t = new ublas::vector<double>(d);
    for (size_t i = 0; i < d; i++)
      (*g_t)(i) = grad_t.getValueAtLocation(i);
    grads.push_back(g_t);
    
    const size_t bs = grads.size(); // The current bundle size.

    // Set the quadratic term to G := A'*A.
    // Note: We could rebuild G from scratch each time (and then add tiny values
    // to the diagonal), but that's much slower, e.g.:
    // ublas::matrix<double> G = ublas::prod(ublas::trans(A), A) / _beta;
    G = copyG;
    if (!dropped) {
      G.resize(bs, bs, true); // Note: true --> preserve existing entries
      const size_t j = bs -1;
      for (size_t i = 0; i < bs; i++) {
        const double temp = ublas::inner_prod(grads[i], grads[j]) / _beta;
        if (i == j)
          G(i, j) = temp + tiny; // Add a small value to diag to ensure pos-def.
        else {
          G(i, j) = temp;
          G(j, i) = temp;
        }
      }
    }
    else {
      assert(!_noShrinking);
      G.resize(bs, bs, false);
      for (size_t i = 0; i < bs; i++) {
        for (size_t j = 0; j <= i; j++) {
          const double temp = ublas::inner_prod(grads[i], grads[j]) / _beta;
          if (i == j)
            G(i, j) = temp + tiny;
          else {
            G(i, j) = temp;
            G(j, i) = temp;
          }
        }
      }
    }
    copyG = G;
    
    // If dropped=true, we already shifted the entries in b during the previous
    // iteration. Therefore, regardless of the value of dropped, we simply have
    // to append the entry corresponding to the current gradient.
    const double b_t = Remp - w.innerProd(grad_t);
    b.resize(bs, true);
    b(bs - 1) = -b_t;
    
    // Encode the constraint: L1-norm(alpha) = 1.
    ublas::matrix<double> CE(bs, 1);
    ublas::vector<double> ce0(1);    
    for (size_t i = 0; i < bs; i++)
      CE(i, 0) = 1;
    ce0(0) = -1;
    
    // Encode the constraint: alpha >= 0.
    ublas::identity_matrix<double> CI(bs);
    ublas::vector<double> ci0(bs);
    for (size_t i = 0; i < bs; i++)
      ci0(i) = 0;
    
    // Allocate a vector to hold the solution.
    ublas::vector<double> alpha(bs);
    
    // Note: solve_quadprog may modify G, which is why we make a copy above.
    double JwCP_ = uQuadProgPP::solve_quadprog(G, b, CE, ce0, CI, ci0, alpha);
    if (JwCP_ == numeric_limits<double>::infinity()) {
      // Signal to the caller that something went wrong...
      return numeric_limits<double>::infinity();
    }
    // Negate the optimal value returned, since BMRM thinks we're maximizing.
    JwCP_ = -JwCP_;
    
    // The qp solution gives us alpha; we need to now compute
    // wTemp = -1/beta*(A*alpha).
    wTemp = alpha(0) * grads[0];
    for (size_t i = 1; i < bs; i++)
      wTemp += alpha(i) * grads[i];
    wTemp /= -_beta;
    
    // Set the entries W in our model to wTemp.
    w.setWeights(wTemp.data().begin(), d);
    
    // Recompute the objective value and gradient.
    _objective.valueAndGradient(w, Remp, grad_t);
    Jw = (0.5 * _beta * w.squaredL2Norm()) + Remp;
    
    // Shrinking heuristic: Discard (grad,b) pairs for which alpha == 0.
    if (!_noShrinking) {
      dropped = false;
      for (int i = bs - 1; i >= 0; i--) {
        if (alpha(i) <= tiny) {
          // Discard the gradient vector
          grads.erase(grads.begin() + i);
          // Shift the entries in b
          for (int j = i; j < bs - 1; j++)
            b(j) = b(j + 1);
          if (!dropped)
            dropped = true;
        }
      }
    }
    
    if (Jw < min_Jw)
      min_Jw = Jw;

    double epsilon_t = min_Jw - JwCP_;
    if (epsilon_t < 0) {
      cout << name() << ": Warning: epsilon_t = " << epsilon_t << " < 0" <<
          " ... Saying converged.\n";
      epsilon_t = 0;
    }
    
    if (!_quiet) {
      printf("%s: t = %d\tJw_t = %0.5e\tepsilon_t = %0.5e\n", name().c_str(),
          (int)t, Jw, epsilon_t);
    }
    
    if (epsilon_t <= tol) {
      if (!_quiet)
        cout << name() << ": Convergence detected; objective value " << min_Jw
          << endl;
      converged = true;
      break;
    }
  }
  
  if (!converged) {
    cout << name() << ": Max iterations reached; objective value " << Jw
      << endl;
  }  
  return min_Jw;
}
