/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentFeatureGen.h"
#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "BmrmOptimizer.h"
#include "CognatePairReader.h"
#include "Dataset.h"
#include "EmOptimizer.h"
#include "EmptyAlignmentFeatureGen.h"
#include "EmptyObservedFeatureGen.h"
#include "Example.h"
#include "InputReader.h"
#include "KlementievRothWordFeatureGen.h"
#include "KlementievRothSentenceFeatureGen.h"
#include "Label.h"
#include "LbfgsOptimizer.h"
#include "LogFeatArc.h"
#include "LogLinearBinary.h"
#include "LogLinearBinaryObs.h"
#include "LogLinearMulti.h"
#include "MaxMarginBinary.h"
#include "MaxMarginBinaryObs.h"
#include "MaxMarginMulti.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Optimizer.h"
#include "Pattern.h"
#include "SentenceAlignmentFeatureGen.h"
#include "SentencePairReader.h"
#include "WordAlignmentFeatureGen.h"
#include "StringEditModel.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "WeightVector.h"
#include "WordPairReader.h"
#include <algorithm>
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
using namespace boost;
using namespace std;

void evaluateMultipleWeightVectors(const vector<WeightVector>&, const Dataset&,
    TrainingObjective&, const string&, int, bool, bool, bool);

int main(int argc, char** argv) {
  // Store the options in string format for later writing to an output file.
  stringstream optsStream;
  for (int i = 1; i < argc; i++)
    optsStream << argv[i] << " ";
  optsStream << endl;
  
  cout << optsStream.str(); // Print the options to stdout.

  // Parse the options.
  namespace opt = boost::program_options;
  bool addBeginEndMarkers = false;
  bool keepAllPositives = false;
  bool noEarlyGridStop = false;
  bool optEM = false;
  bool printAlignments = false;
  bool split = false;
  const string blank("<NONE>");
  const string optAuto("Auto");
  double negativeRatio = 0.0;
  double trainFraction = 1.0;
  int seed = 0;
  size_t threads = 1;
  string dirPath("./");
  string fgenNameLat(blank);
  string fgenNameObs(blank);
  string modelName(blank);
  string objName(blank);
  string optName(blank);
  string readerName(blank);
  string trainFilename(blank);
  string weightsInit(blank);
  vector<double> betas;
  vector<double> tolerances;
  vector<string> evalFilenames;
  
  // Enumerate the choices for each option that involves a class name.
  const string CMA = ", ";
  stringstream fgenMsgLat;
  fgenMsgLat << "feature gen latent {" << EmptyAlignmentFeatureGen::name() <<
      CMA << SentenceAlignmentFeatureGen::name() <<
      CMA << WordAlignmentFeatureGen::name() << "}";  
  stringstream fgenMsgObs;
  fgenMsgObs << "feature gen observed {" << BiasFeatureGen::name() << CMA
      << EmptyObservedFeatureGen::name() << CMA
      << KlementievRothWordFeatureGen::name() << CMA
      << KlementievRothSentenceFeatureGen::name() << "}";      
  stringstream modelMsgObs;
  modelMsgObs << "model {" << StringEditModel<LogFeatArc>::name() << "}";
  stringstream objMsgObs;
  objMsgObs << "obj {" << LogLinearBinary::name() << CMA <<
      LogLinearBinaryObs::name() << CMA << LogLinearMulti::name() << CMA <<
      MaxMarginBinary::name() << CMA <<
      MaxMarginBinaryObs::name() << CMA << MaxMarginMulti::name() << "}";
  stringstream optMsgObs;
  optMsgObs << "opt {" << optAuto << CMA << BmrmOptimizer::name() << CMA <<
      LbfgsOptimizer::name() << "}";
  stringstream readerMsg;
  readerMsg << "reader that parses lines from input file {" <<
      CognatePairReader::name() << CMA << SentencePairReader::name() << CMA <<
      WordPairReader::name() << "}";  
  
  opt::options_description options("Main options");
  options.add_options()
    ("add-begin-end", opt::bool_switch(&addBeginEndMarkers), "add begin-/end-\
of-sequence markers to the examples")
    ("beta,b", opt::value<vector<double> >(&betas)->default_value(
        vector<double>(1, 1.0), "1.0"), "the L2 regularization constant used \
in the training objective, i.e., (beta/2)*||w||^2")
    ("directory", opt::value<string>(&dirPath),
        "directory in which to store results (must already exist)")
//  ("em", opt::bool_switch(&optEM), "employ the optimizer inside an EM loop")
    ("eval", opt::value<vector<string> >(&evalFilenames),
        "evaluation data file")
    ("fgen-latent", opt::value<string>(&fgenNameLat)->default_value(
        EmptyAlignmentFeatureGen::name()), fgenMsgLat.str().c_str())
    ("fgen-observed", opt::value<string>(&fgenNameObs)->default_value(
        EmptyObservedFeatureGen::name()), fgenMsgObs.str().c_str())
    ("keep-all-positives", opt::bool_switch(&keepAllPositives),
        "if --sample-train is enabled, this option ensures that all the \
positive examples present in the data are retained")
    ("model", opt::value<string>(&modelName)->default_value(
        StringEditModel<LogFeatArc>::name()), modelMsgObs.str().c_str())
    ("no-early-grid-stop", opt::bool_switch(&noEarlyGridStop),
        "by default, we break from the grid search loop (over the tolerance \
and beta values) if the optimizer failed to converge; however, if this flag is \
present, all points on the grid will be visited")
    ("objective", opt::value<string>(&objName)->default_value(
        LogLinearMulti::name()), objMsgObs.str().c_str())
    ("optimizer", opt::value<string>(&optName)->default_value(optAuto),
        optMsgObs.str().c_str())
    ("print-alignments", opt::bool_switch(&printAlignments), "print the max-\
scoring alignment for each eval example to a file (requires --eval); note: \
this operation is relatively slow, since it does not make use of multi-\
threading or fst caching")
    ("reader", opt::value<string>(&readerName), readerMsg.str().c_str())
    ("sample-negative-ratio", opt::value<double>(&negativeRatio),
        "for each positive example in the training set, sample this number of \
negative examples (implies --sample-all-positives)")
    ("sample-train", opt::value<double>(&trainFraction),
        "learn on this fraction of the train data (uniformly sampled, \
without replacement); if greater than 1, the value is interpreted as the \
*number* of training examples to sample")
    ("seed", opt::value<int>(&seed)->default_value(0),
        "seed for random number generator")
    ("split", opt::bool_switch(&split), "used in combination with \
sample-train to use the unselected training examples as the eval set")
    ("threads", opt::value<size_t>(&threads)->default_value(1),
        "number of threads employed to parallelize computations")
    ("tolerance,t", opt::value<vector<double> >(&tolerances)->default_value(
        vector<double>(1, 1e-4), "1e-4"), "the tolerance of the stopping \
criterion used by the optimizer")
    ("train", opt::value<string>(&trainFilename), "training data file")
    ("weights-init", opt::value<string>(&weightsInit)->default_value("noise"),
        "initialize weights {heuristic, heuristic+noise, noise, zero}")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  const bool help = vm.count("help");
  const bool writeFiles = vm.count("directory");
  const bool trainFileSpecified = vm.count("train");
  const bool evalFileSpecified = vm.count("eval");
  
  if (!trainFileSpecified && !evalFileSpecified) {
      cout << "Invalid arguments: Either --train or --eval is required\n"
        << options << endl;
    return 1;
  }
  
  if (split && trainFraction == 1.0) {
    cout << "Invalid arguments: Can't have --split with --sample-train=1.0\n"
        << options << endl;
    return 1;
  }
  
  if (!boost::iends_with(dirPath, "/"))
    dirPath += "/";
    
  const string alphabetFname = dirPath + "alphabet.txt";
  bool cachingEnabled = false;
  bool resumed = false; // Are we resuming an incomplete run?
  vector<Model*> models;
  
  shared_ptr<Alphabet> loadedAlphabet(new Alphabet(false, false));
  if (boost::filesystem::exists(alphabetFname)) {
    if (!loadedAlphabet->read(alphabetFname)) {
      cout << "Error: Unable to read " << alphabetFname << endl;
      return 1;
    }
    loadedAlphabet->lock();
    resumed = true;
    cout << "Warning: Found existing output files in " << dirPath <<
        ", so treating this as a resumed run. Any evaluation output files " <<
        "will be overwritten\n";
  }

  for (size_t th = 0; th < threads; th++) {
    shared_ptr<Alphabet> alphabet(new Alphabet(false, false));
    if (resumed)
      alphabet = loadedAlphabet;
    else
      alphabet->unlock(); // in preparation for feature gathering
      
    // initialize the latent feature generator
    shared_ptr<AlignmentFeatureGen> fgenLat;
    if (fgenNameLat == WordAlignmentFeatureGen::name())
      fgenLat.reset(new WordAlignmentFeatureGen(alphabet));
    else if (fgenNameLat == SentenceAlignmentFeatureGen::name())
      fgenLat.reset(new SentenceAlignmentFeatureGen(alphabet));
    else if (fgenNameLat == EmptyAlignmentFeatureGen::name())
      fgenLat.reset(new EmptyAlignmentFeatureGen(alphabet));
    else {
      if (!help) {
        cout << "Invalid arguments: An unrecognized fgen-lat name was given: "
            << fgenNameLat << endl << options << endl;
        return 1;
      }
    }
    if (fgenLat != 0 && fgenLat->processOptions(argc, argv)) {
      cout << "AlignmentFeatureGen::processOptions() failed." << endl;
      return 1;
    }
    
    // initialize the observed feature generator
    shared_ptr<ObservedFeatureGen> fgenObs;
    if (fgenNameObs == BiasFeatureGen::name())
      fgenObs.reset(new BiasFeatureGen(alphabet));
    else if (fgenNameObs == KlementievRothWordFeatureGen::name())
      fgenObs.reset(new KlementievRothWordFeatureGen(alphabet));
    else if (fgenNameObs == KlementievRothSentenceFeatureGen::name())
      fgenObs.reset(new KlementievRothSentenceFeatureGen(alphabet));
    else if (fgenNameObs == EmptyObservedFeatureGen::name())
      fgenObs.reset(new EmptyObservedFeatureGen(alphabet));
    else {
      if (!help) {
        cout << "Invalid arguments: An unrecognized fgen-obs name was given: "
            << fgenNameObs << endl << options << endl;
        return 1;
      }
    }
    if (fgenObs != 0 && fgenObs->processOptions(argc, argv)) {
      cout << "ObservedFeatureGen::processOptions() failed." << endl;
      return 1;
    }
    
    // initialize a model
    Model* model = 0;
    if (modelName == StringEditModel<StdFeatArc>::name()) {
      if (istarts_with(objName, "LogLinear"))
        model = new StringEditModel<LogFeatArc>(fgenLat, fgenObs);
      else if (istarts_with(objName, "MaxMargin"))
        model = new StringEditModel<StdFeatArc>(fgenLat, fgenObs);
      else if (!help) {
        cout << "Invalid arguments: Objective name does not begin with LogLinear "
            << "or MaxMargin: " << objName << endl << options << endl;
        return 1;
      }
    }
    else {
      if (!help) {
        cout << "Invalid arguments: An unrecognized model name was given: "
            << modelName << endl << options << endl;
        return 1;
      }
    }
    if (model != 0 && model->processOptions(argc, argv)) {
      cout << "Model::processOptions() failed." << endl;
      delete model;
      return 1;
    }
    
    if (model != 0) {
      if (!cachingEnabled)
        cachingEnabled = model->getCacheEnabled();
      model->setCacheEnabled(false); // Disable caching during feature gathering    
      models.push_back(model);
    }
  }
  
  boost::timer::auto_cpu_timer timerTotal;
  
  Dataset trainData(threads);
  Dataset evalData(threads);
  
  // initialize the input reader
  scoped_ptr<InputReader> reader;
  if (readerName == CognatePairReader::name())
    reader.reset(new CognatePairReader());
  else if (readerName == SentencePairReader::name())
    reader.reset(new SentencePairReader());
  else if (readerName == WordPairReader::name())
    reader.reset(new WordPairReader());
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized reader name was given: "
          << readerName << endl << options << endl;
      return 1;
    }
  }
  if (reader)
    reader->setAddBeginEndMarkers(addBeginEndMarkers);
  
  if (!help && trainFileSpecified) {
    {
      cout << "Loading " << trainFilename << " ...\n";
      boost::timer::auto_cpu_timer loadTrainTimer;
      if (Utility::loadDataset(*reader, trainFilename, trainData)) {
        cout << "Error: Unable to load train file " << trainFilename << endl;
        return 1;
      }
      cout << "Read " << trainData.numExamples() << " train examples, " <<
          trainData.getLabelSet().size() << " classes\n";
    }
  
    if (negativeRatio > 0.0 || trainFraction != 1.0) {
      if (negativeRatio > 0.0)
        keepAllPositives = true; // keepAllPositives is implied in this case
      mt19937 mt(seed);
      uniform_int<> uni(0, trainData.numExamples()-1);
      variate_generator<mt19937, uniform_int<> > rgen(mt, uni);
      set<int> selected;
      if (keepAllPositives) {
        for (size_t i = 0; i < trainData.numExamples(); i++) {
          if (trainData.getExamples()[i].y() == TrainingObjective::kPositive)
            selected.insert((int)i);
        }
        cout << "Keeping all " << selected.size() << " positive examples.\n";
      }
      size_t subTrainSize;      
      if (negativeRatio > 0.0) {
        subTrainSize = (size_t)(selected.size() * negativeRatio) +
          selected.size();
      }
      else {
        assert(trainFraction != 1.0);
        if (trainFraction < 1)
          subTrainSize = (size_t)(trainFraction * trainData.numExamples());
        else // interpret trainFraction as the number of examples requested
          subTrainSize = (size_t)trainFraction;
      }
      if (subTrainSize >= trainData.numExamples()) {
        cout << "Invalid arguments: There are not enough training examples " <<
            "for the sampling options that you specified.\n";
        return 1;
      }
      if (selected.size() >= subTrainSize) {
        cout << "Invalid arguments: There are more positive examples than " <<
            "the total number of training examples you asked to sample. " <<
            "Please lower the value of --sample-train or remove the " <<
            "--sample-all-positives flag.\n";
        return 1;
      }
      cout << "Randomly selecting " << (subTrainSize - selected.size()) <<
          (keepAllPositives ? " negative" : " train") << " examples.\n";
      while (selected.size() < subTrainSize)
        selected.insert(rgen());
      Dataset subTrainData(threads);
      if (split) {
        // Put selected examples in train, unselected in eval.
        for (size_t i = 0; i < trainData.numExamples(); i++) {
          if (selected.find(i) == selected.end())
            evalData.addExample(trainData.getExamples()[i]);
          else
            subTrainData.addExample(trainData.getExamples()[i]);
        }
        assert(evalData.numExamples() == trainData.numExamples()-subTrainSize);
      }
      else {
        BOOST_FOREACH(int i, selected) {
          subTrainData.addExample(trainData.getExamples()[i]);
        }
      }
      trainData = subTrainData;
    }
  
    if ((int)trainData.getLabelSet().size() <= 1) {
      cout << "Error: Fewer than 2 class labels were found.\n";
      return 1;
    }
    BOOST_FOREACH(const Label y, trainData.getLabelSet()) {
      if (y >= (int)trainData.getLabelSet().size()) {
        cout << "Error: The class labels are not sequential (0,1,...,k).\n";
        return 1;
      }
    }
  }
  
  // Initialize the training objective.
  shared_ptr<TrainingObjective> objective;
  if (objName == LogLinearMulti::name())
    objective.reset(new LogLinearMulti(trainData, models));
  else if (objName == LogLinearBinary::name())
    objective.reset(new LogLinearBinary(trainData, models));
  else if (objName == LogLinearBinaryObs::name())
    objective.reset(new LogLinearBinaryObs(trainData, models));
  else if (objName == MaxMarginBinary::name()) {
    objective.reset(new MaxMarginBinary(trainData, models));
    optEM = true; // EM is currently required for optimizing this objective
  }
  else if (objName == MaxMarginBinaryObs::name())
    objective.reset(new MaxMarginBinaryObs(trainData, models));
  else if (objName == MaxMarginMulti::name()) {
    objective.reset(new MaxMarginMulti(trainData, models));
    optEM = true; // EM is currently required for optimizing this objective
  }
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized objective name was given: "
          << objName << endl << options << endl;
      return 1;
    }
  }
  // Note: If --help is enabled, the data will not have been loaded
  assert(help || threads == objective->getNumModels());

  // Initialize the optimizer.
  shared_ptr<Optimizer> optimizer_;
  if (optName == optAuto) {
    // Automatically select an appropriate optimizer for the chosen objective.
    if (istarts_with(objName, "LogLinear"))
      optimizer_.reset(new LbfgsOptimizer(*objective));
    else if (istarts_with(objName, "MaxMargin"))
      optimizer_.reset(new BmrmOptimizer(*objective));
    else {
      cout << "Automatic optimizer selection failed for " << objName << endl;
      return 1;
    }
  }
  else if (optName == LbfgsOptimizer::name())    
    optimizer_.reset(new LbfgsOptimizer(*objective));
  else if (optName == BmrmOptimizer::name())
    optimizer_.reset(new BmrmOptimizer(*objective));
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized optimizer name was given: "
          << optName << endl << options << endl;
      return 1;
    }
  }
  if (optimizer_->processOptions(argc, argv)) {
    cout << "Optimizer::processOptions() failed." << endl;
    return 1;
  }

  // Wrap the optimizer in an EM procedure if requested.
  shared_ptr<Optimizer> optimizer;
  if (optEM) {
    optimizer.reset(new EmOptimizer(*objective, optimizer_));
    if (optimizer->processOptions(argc, argv)) {
      cout << "Optimizer::processOptions() failed." << endl;
      return 1;
    }
  }
  else
    optimizer = optimizer_; // Note: processOptions has already been called

  if (help) {
    cout << options << endl;
    return 1;
  }
  
  if (!resumed) {
    cout << "Gathering features ...\n";
    size_t maxNumFvs = 0, totalNumFvs = 0;
    {
      boost::timer::auto_cpu_timer gatherTimer;
      objective->gatherFeatures(maxNumFvs, totalNumFvs);
      assert(maxNumFvs > 0 && totalNumFvs > 0);
      objective->combineAndLockAlphabets();
    }
  }
  
  shared_ptr<const AlignmentFeatureGen> fgenLat =
      objective->getModel(0).getFgenLatent();
  shared_ptr<const ObservedFeatureGen> fgenObs =
      objective->getModel(0).getFgenObserved();
  shared_ptr<Alphabet> alphabet = fgenLat->getAlphabet();
  assert(alphabet == fgenObs->getAlphabet()); // Assume the alphabet is shared
  
  const int d = alphabet->size();
  if (d == 0) {
    cout << "Error: No features were found!\n";
    return 1;
  }
  cout << "Extracted " << d << " features\n";  

  // If an output directory was specfied, save the alphabet and options.
  if (writeFiles && !resumed) {
    if (!alphabet->write(alphabetFname))
      cout << "Warning: Unable to write " << alphabetFname << endl;  
    const string optsFname = dirPath + "options.txt";
    ofstream optsOut(optsFname.c_str());
    if (optsOut.good()) {
      optsOut << optsStream.str();
      optsOut.close();
    }
    else
      cout << "Warning: Unable to write " << optsFname << endl;
  }

  // Enable caching at this point, if requested.
  if (cachingEnabled) {
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      objective->getModel(i).setCacheEnabled(true);
      // The cache may contain a "reusable" fst: see StringEditModel::getFst().
      objective->getModel(i).emptyCache();      
    }
  }
  
  assert(betas.size() > 0);
  assert(tolerances.size() > 0);
  WeightVector wInit(d); // zero vector

  // Set initial weights (Note: the reAlloc above set them all to zero).
  if (weightsInit != "zero") {
    double* v = new double[d];
    for (int i = 0; i < d; i++)
      v[i] = 0;    
    if (istarts_with(weightsInit, "heuristic")) {
      const Alphabet::DictType& dict = alphabet->getDict();
      Alphabet::DictType::const_iterator it; 
      for (it = dict.begin(); it != dict.end(); it++)
        v[it->second] += fgenLat->getDefaultFeatureWeight(it->first);
    }
    if (iends_with(weightsInit, "noise")) {
      mt19937 mt(seed);
      normal_distribution<> gaussian(0, 0.01);
      variate_generator<mt19937, normal_distribution<> > rgen(mt, gaussian);
      for (int i = 0; i < d; i++)
        v[i] += rgen();
    }
    wInit.setWeights(v, d);
    delete[] v;
  }

  // Train weights for each combination of the beta and tolerance parameters.
  // Note that the fsts (if caching is enabled) will be reused after being
  // built during the first parameter combination.
  vector<WeightVector> weightVectors;
  sort(tolerances.rbegin(), tolerances.rend()); // sort in descending order
  sort(betas.rbegin(), betas.rend()); // sort in descending order
  BOOST_FOREACH(const double tol, tolerances) {
    BOOST_FOREACH(const double beta, betas) {    
      assert(beta > 0); // by definition, these should be positive values
      assert(tol > 0);
      
      weightVectors.push_back(WeightVector(d));
      WeightVector& w = weightVectors.back();
      w.setWeights(wInit.getWeights(), d);      
      assert(weightVectors.size() > 0);
      const size_t wvIndex = weightVectors.size() - 1;
      
      stringstream weightsFname;
      weightsFname << dirPath << wvIndex << "-weights.txt";
      bool weightsFileIsGood = false;
      
      if (boost::filesystem::exists(weightsFname.str()))
      {
        assert(resumed);
        if (!w.read(weightsFname.str(), alphabet->size()))
          cout << "Warning: Unable to read " << weightsFname.str() << endl;
        else
          weightsFileIsGood = true;
      }
      
      if (!weightsFileIsGood) {
        if (trainFileSpecified) {
          // Train the model.
          Optimizer::status status = Optimizer::FAILURE;
          cout << "Calling Optimizer.train() with beta=" << beta << " and " <<
              "tolerance=" << tol << endl;
          {
            boost::timer::auto_cpu_timer trainTimer;
            double fval = 0.0; // (not used)
            optimizer->setBeta(beta);
            status = optimizer->train(w, fval, tol);
          }        
          if (!noEarlyGridStop && (status == Optimizer::FAILURE || status ==
              Optimizer::MAX_ITERS_CONVEX)) {
            cout << "Warning: Optimizer returned status " << status << ". " <<
                "Discarding classifier and skipping to next tolerance value.\n";
            weightVectors.pop_back();
            break;
          }
          cout << wvIndex << "-status: " << status << endl;
        }
        else {
          // If no train file was specified, we interpret this to mean that the
          // user wishes to evaluate only the existing weight vectors in the
          // given directory.
          cout << "Warning: There will be no results for beta=" << beta <<
              " and tolerance=" << tol << endl;
          weightVectors.pop_back();
          break;
        }
      }
      
      cout << wvIndex << "-beta: " << beta << endl;
      cout << wvIndex << "-tolerance: " << tol << endl;

      // If an output directory was specfied, save the weight vector.
      if (writeFiles && !weightsFileIsGood) {
        if (!w.write(weightsFname.str()))
          cout << "Warning: Unable to write " << weightsFname.str() << endl;
      }
      
      // Classify train examples and optionally write the predictions to a file.
      // Note: The model's transducer cache can still be used at this point, so
      // we defer purging it until after classifying the train examples.
      stringstream predictFname; // Defaults to empty string (for evaluate()).
      if (writeFiles)
        predictFname << dirPath << wvIndex << "-train_predictions.txt";
      if (!boost::filesystem::exists(predictFname.str())) {
        cout << "Classifying train examples ...\n";
        {
          stringstream identifier;
          identifier << wvIndex << "-Train";
          boost::timer::auto_cpu_timer classifyTrainTimer;
          Utility::evaluate(w, *objective, trainData, identifier.str(),
              predictFname.str());
        }
      }
    }
  }
  
  trainData.clear(); // We no longer need the training data.

  // Clear the fsts that were cached for the training data.
  if (cachingEnabled) {
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      Model& model = objective->getModel(i);
      model.setCacheEnabled(false);
      model.emptyCache();
    }
  }
  
  // Load the eval examples from the specified file.
  if (!split && evalFileSpecified && weightVectors.size() > 0) {
    int evalId = 0;
    BOOST_FOREACH(const string& evalFilename, evalFilenames) {
      cout << "Loading " << evalFilename << " ...\n";
      boost::timer::auto_cpu_timer loadEvalTimer;
      Dataset evalData(threads); // Intentionally shadowing previous variable.
      if (Utility::loadDataset(*reader, evalFilename, evalData)) {
        cout << "Error: Unable to load eval file " << evalFilename << endl;
        return 1;
      }
      cout << "Read " << evalData.numExamples() << " eval examples, " <<
          evalData.getLabelSet().size() << " classes\n";
          
      // This situation can arise if, e.g., the eval set only contains examples
      // from one class.
      if (evalData.getLabelSet().size() < trainData.getLabelSet().size()) {
        evalData.addLabels(trainData.getLabelSet());
        cout << "The eval set has fewer classes than train. Adding labels from "
            << "train set to eval set. Number of classes is now "
            << evalData.getLabelSet().size() << ".\n";
      }
      assert(evalData.numExamples() > 0);
      assert(evalData.getLabelSet().size() > 1);
      
      evaluateMultipleWeightVectors(weightVectors, evalData, *objective,
          dirPath, evalId++, writeFiles, printAlignments, cachingEnabled);
    }
  }
  else if (weightVectors.size() == 0) {
    cout << "Warning: No classifiers were successfully trained; therefore, "
        << "no evaluation will be performed. Perhaps the range of the "
        << "tolerance and/or beta parameters should be adjusted. Or, you may "
        << "re-run the experiment with the --no-early-grid-stop flag.\n";
  }

  return 0;
}

// Classify eval examples and optionally write the predictions to files.
// Can also write the alignments to files upon request.
void evaluateMultipleWeightVectors(const vector<WeightVector>& weightVectors,
    const Dataset& evalData, TrainingObjective& objective, const string& path,
    int id, bool writeFiles, bool writeAlignments, bool cachingEnabled) {
  vector<string> identifiers;
  vector<string> fnames;
  for (size_t wvIndex = 0; wvIndex < weightVectors.size(); wvIndex++) {
    stringstream fname;
    if (writeFiles) {
      fname << path << wvIndex << "-eval" << id << "_predictions.txt";
      fnames.push_back(fname.str());
    }
    stringstream identifier;
    identifier << wvIndex << "-Eval";
    identifiers.push_back(identifier.str());
    
    if (writeAlignments) {
      // FIXME: This does not make use of multiple threads or of caching. It
      // should probably be performed alongside the eval predictions. 
      stringstream alignFname;
      alignFname << path << wvIndex << "-eval" << id << "_alignments_yi.txt";
      ofstream alignOut(alignFname.str().c_str());
      if (!alignOut.good()) {
        cout << "Warning: Unable to write " << alignFname.str() << endl;
        continue;
      }
      Model& model = objective.getModel(0);
      assert(!model.getCacheEnabled()); // this would waste memory
      const WeightVector& w = weightVectors[wvIndex];
      cout << "Printing alignments to " << alignFname.str() << ".\n";
      BOOST_FOREACH(const Example& ex, evalData.getExamples()) {
        BOOST_FOREACH(const Label y, evalData.getLabelSet()) {
          if (objective.isBinary() && y != 1)
            continue;
          alignOut << ex.x()->getId() << " (yi = " << ex.y() << ")  y = "
              << y << endl;
          model.printAlignment(alignOut, w, *ex.x(), y);
          alignOut << endl;
        }
      }
      alignOut.close();
    }
  }
  Utility::evaluate(weightVectors, objective, evalData, identifiers, fnames,
      cachingEnabled);
}
