/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentFeatureGen.h"
#include "AlignmentHypergraph.h"
#include "Alphabet.h"
#include "BergsmaKondrakPhrasePairs.h"
#include "BergsmaKondrakWordFeatureGen.h"
#include "BiasFeatureGen.h"
#include "BmrmOptimizer.h"
#include "CognatePairReader.h"
#include "CognatePairAlignerReader.h"
#include "Dataset.h"
#include "EmOptimizer.h"
#include "EmptyAlignmentFeatureGen.h"
#include "EmptyObservedFeatureGen.h"
#include "Example.h"
#include "InputReader.h"
#include "KBestViterbiSemiring.h"
#include "KlementievRothSentenceFeatureGen.h"
#include "KlementievRothWordFeatureGen.h"
#include "Label.h"
#include "LabelScoreTable.h"
#include "LbfgsOptimizer.h"
#include "LogLinearBinary.h"
#include "LogLinearBinaryObs.h"
#include "LogLinearBinaryUnscaled.h"
#include "LogLinearMulti.h"
#include "LogLinearMultiELFV.h"
#include "LogLinearMultiELFV_sigmoid.h"
#include "LogLinearMultiUW.h"
#include "MainHelpers.h"
#include "MaxMarginBinary.h"
#include "MaxMarginBinaryObs.h"
#include "MaxMarginMulti.h"
#include "MaxMarginBinaryPipelineUW.h"
#include "MaxMarginMultiPipelineUW.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Optimizer.h"
#include "Parameters.h"
#include "Pattern.h"
#include "PhrasePairsReader.h"
#include "Regularizer.h"
#include "RegularizerL2.h"
#include "RegularizerNone.h"
#include "RegularizerSoftTying.h"
#include "SentenceAlignmentFeatureGen.h"
#include "SentencePairReader.h"
#include "StochasticGradientOptimizer.h"
#include "StringEditModel.h"
#include "StringPair.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "ValidationSetHandler.h"
#include "WordAlignmentFeatureGen.h"
#include "WordPairReader.h"
#include <algorithm>
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
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

int KBestViterbiSemiring::k;

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
  bool skipTraining = false;
  bool split = false;
  const string blank("<NONE>");
  const string optAuto("Auto");
  double negativeRatio = 0.0;
  double trainFraction = 1.0;
  double weightsNoise;
  int activeLearnIters = 0;
  int samplePool = 0;
  int seed = 0;
  size_t threads = 1;
  string dirPath("./");
  string loadFeaturesFilename;
  string fgenNameLat(blank);
  string fgenNameObs(blank);
  string modelName(blank);
  string objName(blank);
  string optName(blank);
  string readerName(blank);
  string regName(blank);
  string saveFeaturesFilename;
  string trainFilename(blank);
  string validationFilename(blank);
  string weightsInit(blank);
  vector<double> betas;
  vector<double> tolerances;
  vector<string> evalFilenames;
  
  bool stagedTraining = false;
  string uAlphabetFname(blank);
  string wAlphabetFname(blank);
  string uFname(blank);
  string wFname(blank);
  
  // Enumerate the choices for each option that involves a class name.
  const string CMA = ", ";
  stringstream fgenMsgLat;
  fgenMsgLat << "latent feature generator {"
      << EmptyAlignmentFeatureGen::name() <<
      CMA << SentenceAlignmentFeatureGen::name() <<
      CMA << WordAlignmentFeatureGen::name() << "}";  
  stringstream fgenMsgObs;
  fgenMsgObs << "observed feature generator {" << BiasFeatureGen::name() << CMA
      << EmptyObservedFeatureGen::name() << CMA
      << BergsmaKondrakPhrasePairs::name() << CMA
      << BergsmaKondrakWordFeatureGen::name() << CMA
      << KlementievRothWordFeatureGen::name() << CMA
      << KlementievRothSentenceFeatureGen::name() << "}";      
  stringstream modelMsgObs;
  modelMsgObs << "model {" <<
      StringEditModel<AlignmentHypergraph>::name() << "}";
  stringstream objMsgObs;
  objMsgObs << "objective function {" << LogLinearBinary::name() << CMA <<
      LogLinearBinaryUnscaled::name() << CMA <<
      LogLinearBinaryObs::name() << CMA << LogLinearMulti::name() << CMA <<
      LogLinearMultiELFV::name() << CMA << LogLinearMultiELFV_sigmoid::name() <<
      CMA << LogLinearMultiUW::name() << CMA << MaxMarginBinary::name() << CMA <<
      MaxMarginBinaryObs::name() << CMA << MaxMarginMulti::name() << "}";
  stringstream optMsgObs;
  optMsgObs << "optimization algorithm {" << optAuto << CMA <<
      BmrmOptimizer::name() << CMA << LbfgsOptimizer::name() << CMA <<
      StochasticGradientOptimizer::name() << "}";
  stringstream readerMsg;
  readerMsg << "reader that parses lines from input file {" <<
      CognatePairReader::name() << CMA <<
      CognatePairAlignerReader::name() << CMA <<
      PhrasePairsReader::name() << CMA <<
      SentencePairReader::name() << CMA <<
      WordPairReader::name() << "}";  
  stringstream regMsg;
  regMsg << "type of regularization {" << RegularizerL2::name() << CMA <<
      RegularizerSoftTying::name() << CMA << RegularizerNone::name() << "}";
  
  opt::options_description options("Main options");
  options.add_options()
    ("add-begin-end", opt::bool_switch(&addBeginEndMarkers), "add begin-/end-\
of-sequence markers to the examples")
    ("active-learn-iters", opt::value<int>(&activeLearnIters)->default_value(0),
"perform active learning for the given number of iterations (experimental)")
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
    ("k-best-list", opt::value<int>(&KBestViterbiSemiring::k)->default_value(0),
        "extract the k top-scoring latent structures")
    ("keep-all-positives", opt::bool_switch(&keepAllPositives),
        "if --sample-train is enabled, this option ensures that all the \
positive examples present in the data are retained")
    ("load-features", opt::value<string>(&loadFeaturesFilename),
        "load features/alphabet from the given file")
    ("model", opt::value<string>(&modelName)->default_value(
        StringEditModel<AlignmentHypergraph>::name()), modelMsgObs.str().c_str())
    ("no-early-grid-stop", opt::bool_switch(&noEarlyGridStop),
        "by default, we break from the grid search loop (over the tolerance \
and beta values) if the optimizer failed to converge; however, if this flag is \
present, all points on the grid will be visited")
    ("no-training", opt::bool_switch(&skipTraining), "load and evaluate \
any existing weight vectors, but don't train at any new points in the grid")
    ("objective", opt::value<string>(&objName)->default_value(
        LogLinearMulti::name()), objMsgObs.str().c_str())
    ("optimizer", opt::value<string>(&optName)->default_value(optAuto),
        optMsgObs.str().c_str())
    ("print-alignments", opt::bool_switch(&printAlignments), "print the max-\
scoring alignment for each eval example to a file (requires --eval); note: \
this operation is relatively slow, since it does not make use of multi-\
threading or fst caching")
    ("reader", opt::value<string>(&readerName), readerMsg.str().c_str())
    ("regularizer", opt::value<string>(&regName)->default_value(
        RegularizerL2::name()), regMsg.str().c_str())
    ("sample-negative-ratio", opt::value<double>(&negativeRatio),
        "for each positive example in the training set, sample this number of \
negative examples (implies --keep-all-positives)")
    ("sample-pool", opt::value<int>(&samplePool)->default_value(0),
        "in active learning mode, randomly sample a sub-pool of the given size \
prior to selective sampling")
    ("sample-train", opt::value<double>(&trainFraction),
        "learn on this fraction of the train data (uniformly sampled, \
without replacement); if greater than 1, the value is interpreted as the \
*number* of training examples to sample")
    ("save-features", opt::value<string>(&saveFeaturesFilename),
        "save features/alphabet to the given file")
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
    ("validation", opt::value<string>(&validationFilename),
        "validation data file (only used by certain optimizers)")
    ("weights-init", opt::value<string>(&weightsInit)->default_value("noise"),
        "initialize weights {heuristic, heuristic+noise, noise, zero}")
    ("weights-noise-level", opt::value<double>(&weightsNoise)->default_value(
        0.01), "Gaussian variance parameter used when applying noise to \
initial weights")
    ("help", "display a help message")
    
    // The following are used only by staged training and u-w pipeline demo.
    ("staged-training", opt::bool_switch(&stagedTraining),
        "perform staged training by loading the w-alphabet and w-weights, then \
initializing w based on the loaded model (other features are initialized \
according to weights-init")
    ("u-alphabet", opt::value<string>(&uAlphabetFname)->default_value(
        "<u alphabet filename>"), "file from which to load the u alphabet")
    ("w-alphabet", opt::value<string>(&wAlphabetFname)->default_value(
        "<w alphabet filename>"), "file from which to load the w alphabet")
    ("u-weights", opt::value<string>(&uFname)->default_value(
        "<u weights filename>"), "file from which to load the u weight vector")
    ("w-weights", opt::value<string>(&wFname)->default_value(
        "<w weights filename>"), "file from which to load the w weight vector")
  ;
  
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  const bool help = vm.count("help");
  const bool writeFiles = vm.count("directory");
  const bool trainFileSpecified = vm.count("train");
  const bool evalFileSpecified = vm.count("eval");
  const bool validationFileSpecified = vm.count("validation");
  
  /* Certain training objectives represent two-layer pipeline classifiers. Note
   * that you can only use an existing model for classification at this time, as
   * training the pipeline is not yet supported.
   * Example options are: --eval=data/FVInput/gr-en.prepared.0.58.test
   * --model=StringEdit --obj=MaxMarginBinaryPipelineUW --reader=CognatePair
   * --fgen-lat=WordAlignment --order=0 --no-normalize --no-state-ngrams
   * --exact-match-state --fgen-obs=BKWord --substring-size=3 --add-begin-end
   * --bk-ned --w-alphabet=gr-alphabet_w.txt --w-weights=gr-weights_w.txt
   * --u-weights=gr-weights_u.txt --u-alphabet=gr-alphabet_u.txt
   * The input files for the u model can be created using the script
   * make_alignment_alphabet.py
   */
  const bool pipeline = (objName == MaxMarginBinaryPipelineUW::name()) ||
      (objName == MaxMarginMultiPipelineUW::name());
  
  bool loadFeatures = loadFeaturesFilename.size() > 0;
  bool saveFeatures = saveFeaturesFilename.size() > 0;
  if (pipeline) {
    loadFeatures = true;
    saveFeatures = false;
  }

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
  
  if (istarts_with(objName, "MaxMargin") && regName != RegularizerL2::name()) {
    cout << "Invalid arguments: MaxMargin objectives must use L2 " <<
        "regularization at this time.\n" << options << endl;
    return 1;
  }
  
  if (fgenNameLat == EmptyAlignmentFeatureGen::name() &&
      fgenNameObs == EmptyObservedFeatureGen::name()) {
    cout << "Invalid arguments: fgen-latent and fgen-observed are both set " <<
        "to empty feature generators.\n";
    return 1;
  }
  
  if (loadFeatures && saveFeatures) {
    cout << "Invalid arguments: Can't use --load-features and " <<
        "--save-features at the same time.\n";
    return 1;
  }
  
  if (!iends_with(dirPath, "/"))
    dirPath += "/";
  
  bool cachingEnabled = false;
  bool resumed = false; // Are we resuming an incomplete run?
  vector<Model*> models;
  shared_ptr<Alphabet> loadedAlphabet;
  shared_ptr<Alphabet> wAlphabet;
  string alphabetFname;
  
  if (!pipeline) {
    alphabetFname = dirPath + "alphabet.txt"; 
    
    loadedAlphabet.reset(new Alphabet(false, false));
    if (filesystem::exists(alphabetFname)) {
      if (loadFeatures || saveFeatures) {
        cout << "Error: Can't load or save features when the default " <<
            "named alphabet file (" << alphabetFname << ") already exists\n";
        return 1;
      }
      if (!loadedAlphabet->read(alphabetFname)) {
        cout << "Error: Unable to read " << alphabetFname << endl;
        return 1;
      }
      resumed = true;
      cout << "Warning: Found existing output files in " << dirPath <<
          ", so treating this as a resumed run. Any evaluation output files " <<
          "will be overwritten.\n";
    }
  }
  else {
    // Load two alphabets from files, one each for the w and u parameters.
    if (!filesystem::exists(uAlphabetFname)) {
      cout << "Error: " << uAlphabetFname << " does not exist.\n";
      return 1;
    }
    if (!filesystem::exists(wAlphabetFname)) {
      cout << "Error: " << wAlphabetFname << " does not exist.\n";
      return 1;
    }
    wAlphabet.reset(new Alphabet(false, false));
    if (!wAlphabet->read(wAlphabetFname)) {
      cout << "Error: Unable to read " << wAlphabetFname << endl;
      return 1;
    }
    cout << "Loaded w alphabet from " << wAlphabetFname << endl;
      
    Alphabet uAlphabet(false, false);
    if (!uAlphabet.read(uAlphabetFname)) {
      cout << "Error: Unable to read " << uAlphabetFname << endl;
      return 1;
    }
    cout << "Loaded u alphabet from " << uAlphabetFname << endl;
    
    // Append the u features to the w alphabet.
    for (int i = 0; i < uAlphabet.numFeaturesPerClass(); i++) {
      string f = uAlphabet.reverseLookup(i);
      wAlphabet->lookup(f, TrainingObjective::kPositive, true);
    }
    wAlphabet->lock();
  }

  // The pipeline models may need to lock and unlock the alphabets more than
  // once, since alphabets are loaded, then additional feature gathering may
  // be performed. For convenience, we store a pointer to the alphabet for each
  // thread.
  vector<shared_ptr<Alphabet> > threadAlphabets;
  
  for (size_t th = 0; th < threads; th++) {
    shared_ptr<Alphabet> alphabet;
    if (pipeline) {
      // Each thread needs its own copy, since we may be gathering additional
      // features from the training data.
      alphabet.reset(new Alphabet(*wAlphabet));
      threadAlphabets.push_back(alphabet);
    }
    else if (resumed)
      alphabet = loadedAlphabet;
    else {
      // Each thread gets an empty Alphabet; we combine them below.
      alphabet.reset(new Alphabet(false, false));
    }
      
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
    else if (fgenNameObs == BergsmaKondrakPhrasePairs::name())
      fgenObs.reset(new BergsmaKondrakPhrasePairs(alphabet));
    else if (fgenNameObs == BergsmaKondrakWordFeatureGen::name())
      fgenObs.reset(new BergsmaKondrakWordFeatureGen(alphabet));
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
    if (modelName == StringEditModel<AlignmentHypergraph>::name()) {
      model = new StringEditModel<AlignmentHypergraph>(fgenLat, fgenObs);
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
  
  timer::auto_cpu_timer timerTotal;
  
  Dataset trainData(threads);
  Dataset evalData(threads);
  
  // initialize the input reader
  scoped_ptr<InputReader> reader;
  if (readerName == CognatePairReader::name())
    reader.reset(new CognatePairReader());
  else if (readerName == CognatePairAlignerReader::name())
    reader.reset(new CognatePairAlignerReader());
  else if (readerName == PhrasePairsReader::name())
    reader.reset(new PhrasePairsReader());
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
      timer::auto_cpu_timer loadTrainTimer;
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
  else if (objName == LogLinearMultiELFV::name())
    objective.reset(new LogLinearMultiELFV(trainData, models));
  else if (objName == LogLinearMultiELFV_sigmoid::name())
    objective.reset(new LogLinearMultiELFV_sigmoid(trainData, models));
  else if (objName == LogLinearMultiUW::name())
    objective.reset(new LogLinearMultiUW(trainData, models));
  else if (objName == LogLinearBinary::name())
    objective.reset(new LogLinearBinary(trainData, models));
  else if (objName == LogLinearBinaryUnscaled::name())
    objective.reset(new LogLinearBinaryUnscaled(trainData, models));
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
  else if (objName == MaxMarginBinaryPipelineUW::name()) {
    objective.reset(new MaxMarginBinaryPipelineUW(trainData, models));
    optEM = true;
  }
  else if (objName == MaxMarginMultiPipelineUW::name()) {
    objective.reset(new MaxMarginMultiPipelineUW(trainData, models));
    optEM = true;
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

  // Initialize the regularizer.
  shared_ptr<Regularizer> regularizer;
  if (regName == RegularizerNone::name())
    regularizer.reset(new RegularizerNone());
  else if (regName == RegularizerL2::name())
    regularizer.reset(new RegularizerL2());
  else if (regName == RegularizerSoftTying::name())
    regularizer.reset(new RegularizerSoftTying());
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized regularizer name was given: "
          << regName << endl << options << endl;
      return 1;
    }
  }
  if (regularizer->processOptions(argc, argv)) {
    cout << "Regularizer::processOptions() failed." << endl;
    return 1;
  }

  // Initialize the optimizer.
  shared_ptr<Optimizer> optInner;
  if (optName == optAuto) {
    // Automatically select an appropriate optimizer for the chosen objective.
    if (istarts_with(objName, "LogLinear"))
      optInner.reset(new LbfgsOptimizer(objective, regularizer));
    else if (istarts_with(objName, "MaxMargin"))
      optInner.reset(new BmrmOptimizer(objective, regularizer));
    else {
      cout << "Automatic optimizer selection failed for " << objName << endl;
      return 1;
    }
  }
  else if (optName == LbfgsOptimizer::name())    
    optInner.reset(new LbfgsOptimizer(objective, regularizer));
  else if (optName == BmrmOptimizer::name())
    optInner.reset(new BmrmOptimizer(objective, regularizer));
  else if (optName == StochasticGradientOptimizer::name())
    optInner.reset(new StochasticGradientOptimizer(objective, regularizer));
  else {
    if (!help) {
      cout << "Invalid arguments: An unrecognized optimizer name was given: "
          << optName << endl << options << endl;
      return 1;
    }
  }
  if (optInner->processOptions(argc, argv)) {
    cout << "Optimizer::processOptions() failed." << endl;
    return 1;
  }
  
  boost::shared_ptr<Dataset> validationData;
  shared_ptr<ValidationSetHandler> vsh;
  if (!help && validationFileSpecified) {
    // Enumerate the examples in the validation set such that the ids do not
    // overlap with the training set.
    const size_t nextId = trainData.getMaxId() + 1;
    validationData.reset(new Dataset(threads));
    cout << "Loading " << validationFilename << " ...\n";
    timer::auto_cpu_timer loadValidationTimer;
    if (Utility::loadDataset(*reader, validationFilename, *validationData,
        nextId)) {
      cout << "Error: Unable to load validation file " << validationFilename <<
          endl;
      return 1;
    }
    assert(nextId == validationData->getExamples()[0].x()->getId());
    cout << "Read " << validationData->numExamples() << " validation examples, "
        << validationData->getLabelSet().size() << " classes\n";
    vsh.reset(new ValidationSetHandler(validationData, objective));
    if (vsh->processOptions(argc, argv)) {
      cout << "ValidationSetHandler::processOptions() failed." << endl;
      return 1;
    }
  }

  // Wrap the optimizer in an EM procedure if requested.
  shared_ptr<Optimizer> optimizer;
  if (optEM) {
    optimizer.reset(new EmOptimizer(objective, regularizer, optInner));
    if (optimizer->processOptions(argc, argv)) {
      cout << "Optimizer::processOptions() failed." << endl;
      return 1;
    }
  }
  else
    optimizer = optInner; // Note: processOptions has already been called
    
  optimizer->setValidationSetHandler(vsh); // still OK if vsh is null

  if (help) {
    cout << options << endl;
    return 1;
  }
  
  if (!resumed && !loadFeatures) {
    cout << "Gathering features ...\n";
    size_t maxNumFvs = 0, totalNumFvs = 0;
    {
      timer::auto_cpu_timer gatherTimer;
      objective->gatherFeatures(maxNumFvs, totalNumFvs);
      assert(maxNumFvs > 0 && totalNumFvs > 0);
      assert(trainData.getLabelSet().size() > 0);
      objective->combineAlphabets(trainData.getLabelSet());
    }
  }
  
  shared_ptr<const AlignmentFeatureGen> fgenLat =
      objective->getModel(0).getFgenLatent();
  shared_ptr<const ObservedFeatureGen> fgenObs =
      objective->getModel(0).getFgenObserved();
  shared_ptr<Alphabet> alphabet = fgenLat->getAlphabet();
  assert(alphabet == fgenObs->getAlphabet()); // Assume the alphabet is shared

  if (!pipeline) {
    if (loadFeatures) {
      if (!filesystem::exists(loadFeaturesFilename)) {
        cout << "Error: " << loadFeaturesFilename << " does not exist.\n";
        return 1;
      }
      if (!alphabet->read(loadFeaturesFilename)) {
        cout << "Error: Unable to read " << loadFeaturesFilename << endl;
        return 1;
      }
      cout << "Loaded features from " << loadFeaturesFilename << endl;
      assert(trainData.getLabelSet().size() > 0);
      alphabet = objective->combineAlphabets(trainData.getLabelSet());
    }
    
    if (alphabet->size() == 0) {
      cout << "Error: No features were found!\n";
      return 1;
    }
    cout << "Extracted " << alphabet->size() << " features\n";  
    
    if (saveFeatures) {
      alphabet->lock();
      if (!alphabet->write(saveFeaturesFilename)) {
        cout << "Warning: Unable to write " << saveFeaturesFilename << endl;
        return 1;
      }
      else {
        cout << "Saved features to " << saveFeaturesFilename << endl;
        return 0;
      }
    }
  }

  // Enable caching at this point, if requested.
  if (cachingEnabled) {
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      objective->getModel(i).setCacheEnabled(true);
      // The cache may contain a "reusable" fst: see StringEditModel::getFst().
      objective->getModel(i).emptyCache();
      if (optimizer->isOnline() && validationData->numExamples() > 0) {
        // If we're employing an online learner (e.g., SGD), we don't want to
        // cache graphs for training examples because presumably the dataset is
        // very large. In this case, if caching is enabled, we interpret it to
        // mean that we want to cache the validation set examples, which will
        // be classified multiple times during training.
        objective->getModel(i).onlyCacheIdsGreaterThanOrEqualTo(
            validationData->getExamples()[0].x()->getId());
      }
    }
  }
  
  assert(betas.size() > 0);
  assert(tolerances.size() > 0);
  
  // Set the initial parameters.
  Parameters theta0 = objective->getDefaultParameters(alphabet->size());
  // Note: The call to setupParameters may modify theta0 and alphabet, such
  // that, e.g., alphabet->size() may subsequently return a different value.
  // FIXME: We cannot add another dummy label if an alphabet was loaded that
  // already had a dummy label added to it.
  if (resumed && regName == RegularizerSoftTying::name()) {
    cout << "Error: Resuming an experiment with the " << regName <<
        "regularizer is not supported in this version.\n";
    return 1;
  }
  // A label that is "instantiated" has at least one feature associated with it.
  // Specifically, for a binary objective, the negative class will not be
  // instantiated.
  set<Label> instantiatedLabels;
  if (objective->isBinary())
    instantiatedLabels.insert(TrainingObjective::kPositive);
  else
    instantiatedLabels = trainData.getLabelSet();
  regularizer->setupParameters(theta0, *alphabet, instantiatedLabels,
      seed);
  
  if (pipeline) {
    assert(theta0.hasU());
    if (!filesystem::exists(uFname)) {
      cout << "Error: " << uFname << " does not exist.\n";
      return 1;
    }
    if (!filesystem::exists(wFname)) {
      cout << "Error: " << wFname << " does not exist.\n";
      return 1;
    }
    assert(threads == threadAlphabets.size());
    BOOST_FOREACH(shared_ptr<Alphabet>& a, threadAlphabets) {
      if (alphabet->size() != a->size()) {
        cout << "Error: Alphabets for different threads differ in size.\n";
        assert(0);
      }
      a->lock();
    }
    theta0.w.read(wFname, alphabet->size());
    cout << "Loaded w weights from " << wFname << endl;
    theta0.u.read(uFname, alphabet->size());
    cout << "Loaded u weights from " << uFname << endl;
    assert(theta0.w.getDim() == theta0.u.getDim());
    
    // Since the w and u features are assumed to be non-overlapping, the w
    // vector should not contain a non-zero weight where the u vector also has
    // a non-zero weight. We perform the check here as a way of possibly
    // warning the user that they have loaded incompatible weight vectors.
    for (size_t i = 0; i < theta0.w.getDim(); i++) {
      double w = theta0.w[i];
      double u = theta0.u[i];
      if (w != 0 && u != 0) {
        cout << "Warning: The loaded w and u vectors both have non-zero " <<
            "weights for coordinate " << i << " (and possibly others)\n";
        assert(0);
        break;
      }
    }
    
    if (KBestViterbiSemiring::k > 0) {
      assert(alphabet->isLocked());
      if (trainFileSpecified) {
        cout << "Generating " << KBestViterbiSemiring::k << "-best " <<
            "lists (training data) ...\n";
        timer::auto_cpu_timer timer;
        objective->clearKBest();
        objective->initKBest(trainData, theta0);
      }
      if (validationFileSpecified) {
        assert(validationData);
        cout << "Generating " << KBestViterbiSemiring::k << "-best " <<
            "lists (validation data) ...\n";
        timer::auto_cpu_timer timer;
        objective->clearKBest();
        objective->initKBest(*validationData, theta0);
      }
    }
    
    if (trainFileSpecified) {
      shared_ptr<Alphabet> uwAlphabet(new Alphabet(*alphabet));
      const size_t oldSize = uwAlphabet->size();
      BOOST_FOREACH(shared_ptr<Alphabet>& a, threadAlphabets) {
        assert(a->isLocked());
        a->unlock();
      }
      cout << "Gathering additional features from k-best lists ...\n";
      {
        size_t maxNumFvs = 0, totalNumFvs = 0;
        timer::auto_cpu_timer gatherTimer;
        objective->gatherFeatures(maxNumFvs, totalNumFvs);
        assert(maxNumFvs > 0 && totalNumFvs > 0);
      }
      assert(trainData.getLabelSet().size() > 0);
      alphabet = objective->combineAlphabets(trainData.getLabelSet());
      alphabet->lock();      
      threadAlphabets.clear();

      const size_t newSize = alphabet->size();
      if (newSize > oldSize) {
        // If we added some new features, we need to grow w and u to match the
        // new alphabet size.
        
        // Append the newly extracted feature to the ones we initially loaded.
        shared_ptr<Alphabet> uwAlphabetOld(new Alphabet(*uwAlphabet));
        uwAlphabet->unlock();
        for (int i = 0; i < alphabet->numFeaturesPerClass(); i++) {
          string f = alphabet->reverseLookup(i);
          uwAlphabet->lookup(f, TrainingObjective::kPositive, true);
        }
        uwAlphabet->lock();
        
        // We will copy w and u into larger weight vectors, which are set to
        // zero initially.
        assert(oldSize == theta0.getDimW());
        assert(oldSize == theta0.getDimU());
        Parameters temp(oldSize, oldSize);
        temp.setParams(theta0);        
        theta0 = Parameters(newSize, newSize);

        // Remap the old/loaded weights so they correspond to the new (expanded)
        // alphabet. The new weights are simply set to zero.
        BOOST_FOREACH(const Label y, trainData.getLabelSet()) {
          for (int i = 0; i < uwAlphabetOld->numFeaturesPerClass(); i++) {
            const string feature = uwAlphabetOld->reverseLookup(i);
            const int oldIndex = uwAlphabetOld->lookup(feature, y, false);
            const int newIndex = uwAlphabet->lookup(feature, y, false);
            theta0.w.setWeight(newIndex, temp.w[oldIndex]);
            theta0.u.setWeight(newIndex, temp.u[oldIndex]);
          }
        }        
        alphabet = uwAlphabet; // the old alphabet can be overwritten
        for (size_t i = 0; i < objective->getNumModels(); i++) {
          objective->getModel(i).getFgenLatent()->setAlphabet(alphabet);
          objective->getModel(i).getFgenObserved()->setAlphabet(alphabet);
        }
        
        // Clear any feature vectors that have been cached.
        if (cachingEnabled)
          for (size_t i = 0; i < objective->getNumModels(); i++)
            objective->getModel(i).emptyCache();
        
        // Finally, we now need to regenerate the k-best lists, since the
        // max feature vectors were invalidated when we modified the alphabet.
        if (KBestViterbiSemiring::k > 0) {
          assert(alphabet->isLocked());
          if (trainFileSpecified) {
            cout << "Re-generating " << KBestViterbiSemiring::k << "-best " <<
                "lists with additional features (training data) ...\n";
            timer::auto_cpu_timer timer;
            objective->clearKBest();
            objective->initKBest(trainData, theta0);
          }
          if (validationFileSpecified) {
            cout << "Re-generating " << KBestViterbiSemiring::k << "-best " <<
                "lists with additional features (validation data) ...\n";
            assert(validationData);
            timer::auto_cpu_timer timer;
            objective->clearKBest();
            objective->initKBest(*validationData, theta0);
          }
        }
      }
    }
  }
  else {
    alphabet->lock(); // We can lock the Alphabet at this point.
    initWeights(theta0.w, weightsInit, weightsNoise, seed, alphabet,
        instantiatedLabels, fgenLat);
    if (theta0.hasU()) {
      // Note: We modify the seed to avoid symmetry in the parameters.
      initWeights(theta0.u, weightsInit, weightsNoise, seed + 1, alphabet,
          instantiatedLabels, fgenLat);
    }
  
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
  }

  if (activeLearnIters > 0) {
    // Let train be the seed set, eval be the pool, and validation (optional)
    // be the dev (and there's no holdout test).
    // The seed set should contain all the positives and one randomly chosen
    // negative.
    // The eval set should contain the remaining negatives.

    const double beta = betas.front();
    const double tol = tolerances.front();
    const string poolFilename = evalFilenames.front();
    regularizer->setBeta(beta);
    
    Dataset poolData(threads);
    const size_t nextId = max(validationData ? validationData->getMaxId() : 0,
        trainData.getMaxId()) + 1;
    if (Utility::loadDataset(*reader, poolFilename, poolData, nextId)) {
      cout << "Error: Unable to load eval file " << poolFilename << endl;
      return 1;
    }
    cout << "Read " << poolData.numExamples() << " eval examples, " <<
        poolData.getLabelSet().size() << " classes\n";

    for (int iter = 0; iter < activeLearnIters; iter++) {
      if (poolData.numExamples() == 0) {
        cout << "There are no examples left in the pool! Terminating.\n";
        break;
      }
      
      // train on trainData (early stopping, and thus fval, based on
      // validationData)
      // note: validationData will already be setup to use; no need to pass it
      // explicitly here
      {
        cout << "==== AL: Training\n";
        theta0.zero();
        timer::auto_cpu_timer timer;
        double fval = 0.0;
        Optimizer::status status = optimizer->train(theta0, fval, tol);
        if (status == Optimizer::FAILURE) {
          cout << "Optimizer returned FAILURE status. Breaking from AL loop.\n";
          break;
        }
      }
      
      // call TrainingObjective::predict on poolData
      // note: we use the trainData label set, since the pool contains only
      // negative examples
      cout << "==== AL: Predicting on pool data\n";
      Dataset subPoolData(threads);
      LabelScoreTable scores(poolData.getMaxId() + 1,
          trainData.getLabelSet().size());
      {        
        timer::auto_cpu_timer timer;
        if (samplePool == 0 || samplePool >= poolData.numExamples()) {
          subPoolData = poolData; // FIXME: expensive copy
        }
        else {
          cout << "==== AL: Drawing " << samplePool << " examples from pool\n";
          assert(samplePool > 0);
          shared_array<int> s = Utility::randPerm(poolData.numExamples(),
              seed + iter);
          for (int i = 0; i < samplePool; i++)
            subPoolData.addExample(poolData.getExamples()[s[i]]);
        }
        objective->predict(theta0, subPoolData, scores);
      }
      
      // find the most positive (highest score for label 1?) prediction x on
      // poolData
      cout << "==== AL: Performing selective sampling\n";
      const Example* maxScoreExample = &subPoolData.getExamples()[0];
      double maxScore = scores.getScore(maxScoreExample->x()->getId(),
          TrainingObjective::kPositive);
      {
        timer::auto_cpu_timer timer;
        vector<Example>::const_iterator it = subPoolData.getExamples().begin();
        ++it; // skip the first example
        for (; it != subPoolData.getExamples().end(); ++it) {
          size_t id = it->x()->getId();
          double score = scores.getScore(id, TrainingObjective::kPositive);
//        cout << score << " " << (const StringPair&)(*it->x()) << endl;
          if (score > maxScore) {
            maxScore = score;
            maxScoreExample = &(*it);
          }
        }
        const double scoreNegativeLabel = scores.getScore(
            maxScoreExample->x()->getId(), !TrainingObjective::kPositive);
        cout << (const StringPair&)(*maxScoreExample->x()) << " " << maxScore <<
            " " << scoreNegativeLabel << endl;
      }

      // add x to trainData, remove x from poolData
      {
        cout << "==== AL: Updating train and pool data\n";
        timer::auto_cpu_timer timer;
        trainData.addExample(*maxScoreExample);
        Dataset newPoolData(threads);
        vector<Example>::const_iterator it = poolData.getExamples().begin();
        for (; it != poolData.getExamples().end(); ++it) {
          // skip the selected example
          if (it->x()->getId() != maxScoreExample->x()->getId())
            newPoolData.addExample(*it);
        }
        poolData = newPoolData;
        cout << "The pool size is now " << poolData.numExamples() << "; " <<
            "the training size is " << trainData.numExamples() << endl;
      }
      
      // The new example x may contain features we haven't seen before, so
      // we need to update the alphabet and parameters. We simply recreate them
      // from scratch here for the sake of convenience.
      // If we did not load (presumably) all the features that will be available
      // in training data (i.e., train + pool), then we need to "grow" the
      // feature set and parameter vector as we proceed.
      if (!loadFeatures)
      {
        cout << "==== AL: Updating alphabet and parameters\n";
        timer::auto_cpu_timer timer;
        alphabet->unlock();
        {
          size_t maxNumFvs = 0, totalNumFvs = 0;
          objective->gatherFeatures(maxNumFvs, totalNumFvs);
          assert(maxNumFvs > 0 && totalNumFvs > 0);
          alphabet = objective->combineAlphabets(trainData.getLabelSet());
        }
        alphabet->lock();
        theta0 = objective->getDefaultParameters(alphabet->size());
        // note: theta0 is set to zero (i.e., all zeros) at this point
        assert(theta0.w.getDim() == alphabet->size());
      }
    }    
    return 0;
  }
  
  // If we're performing staged training (i.e., we have already trained a
  // presumably "simpler" classifier and now we want to add more features),
  // then we initialize the model we're about to train with the weights of the
  // model from the previous stage. Features that are unique to the current
  // model are initialized according to the --weights-init option.
  if (stagedTraining) {
    assert(!theta0.hasU()); // u-w models do not yet support staged training
    shared_ptr<Alphabet> alphabetPrevStage(new Alphabet(false, false));
    if (!alphabetPrevStage->read(wAlphabetFname)) {
      cout << "Error: Unable to read " << wAlphabetFname << endl;
      return 1;
    }    
    WeightVector wPrevStage(alphabetPrevStage->size());
    wPrevStage.read(wFname, alphabetPrevStage->size());
    cout << "Staged training: Initializing weights from " << wFname <<
        " using " << wAlphabetFname << endl;
    Alphabet::DictType::const_iterator it = alphabetPrevStage->getDict().begin();
    for (; it != alphabetPrevStage->getDict().end(); ++it) {
      string f = it->first;
      int indexPrevStage = it->second;
      assert(indexPrevStage >= 0);
      BOOST_FOREACH(Label y, alphabetPrevStage->getUniqueLabels()) {
        int index = alphabet->lookup(f, y, false);
        if (index >= 0)
          theta0.w.setWeight(index, wPrevStage[indexPrevStage]);
      }
    }
  }
  
  // Train weights for each combination of the beta and tolerance parameters.
  // Note that the fsts (if caching is enabled) will be reused after being
  // built during the first parameter combination.
  vector<Parameters> weightVectors;
  sort(tolerances.rbegin(), tolerances.rend()); // sort in descending order
  sort(betas.rbegin(), betas.rend()); // sort in descending order
  BOOST_FOREACH(const double tol, tolerances) {
    BOOST_FOREACH(const double beta, betas) {    
      assert(beta > 0); // by definition, these should be positive values
      assert(tol > 0);
      
      weightVectors.push_back(objective->getDefaultParameters(
          alphabet->size()));

      Parameters& theta = weightVectors.back();
      theta.setParams(theta0);
      assert(weightVectors.size() > 0);
      const size_t wvIndex = weightVectors.size() - 1;
      
      stringstream weightsFnameW, weightsFnameU;
      weightsFnameW << dirPath << wvIndex << "-weights.txt";
      weightsFnameU << dirPath << wvIndex << "-weightsU.txt";
      bool weightsFileIsGood = false;
      
      if (filesystem::exists(weightsFnameW.str()))
      {
        assert(resumed);
        if (theta.w.read(weightsFnameW.str(), alphabet->size()))
          weightsFileIsGood = true;
        else
          cout << "Warning: Unable to read " << weightsFnameW.str() << endl;
          
        if (weightsFileIsGood && theta.hasU())
        {
          if (!theta.u.read(weightsFnameU.str(), alphabet->size())) {
            cout << "Warning: Unable to read " << weightsFnameU.str() << endl;
            weightsFileIsGood = false;
          }
        }
      }      
      
      if (!weightsFileIsGood) {
        if (trainFileSpecified && !skipTraining) {
          // Train the model.
          Optimizer::status status = Optimizer::FAILURE;
          double fval = 0.0;
          cout << "Calling Optimizer.train() with beta=" << beta << " and " <<
              "tolerance=" << tol << endl;
          {
            timer::auto_cpu_timer trainTimer;
            regularizer->setBeta(beta);
            status = optimizer->train(theta, fval, tol);
          }
          // Update: Besides the cost of evaluation, there's no drawback to
          // keeping classifiers for which the optimizer didn't report
          // convergence. The status still gets reported below.
//        if (status == Optimizer::FAILURE) {
//          cout << "Warning: Optimizer returned status " << status << ". " <<
//              "Discarding classifier.\n";
//          weightVectors.pop_back();
//          continue;
//        }
          if (!noEarlyGridStop && status == Optimizer::MAX_ITERS_CONVEX) {
            cout << "Warning: Optimizer returned status " << status << ". " <<
                "Discarding classifier and skipping to next tolerance value.\n";
            weightVectors.pop_back();
            break;
          }
          cout << wvIndex << "-status: " << status << endl;
          cout << wvIndex << "-Objective-Value: " << fval << endl;
        }
        else {
          // If no train file was specified or the user requested that training
          // be skipped, we evaluate only the existing weight vectors in the
          // given directory (but do not train any new ones, even if there are
          // points on the grid that have not been successfully tried).
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
        if (!theta.w.write(weightsFnameW.str()))
          cout << "Warning: Unable to write " << weightsFnameW.str() << endl;
        if (!theta.u.write(weightsFnameU.str()))
          cout << "Warning: Unable to write " << weightsFnameU.str() << endl;
      }
      
      // Classify train examples and optionally write the predictions to a file.
      // Note: The model's transducer cache can still be used at this point, so
      // we defer purging it until after classifying the train examples.
      stringstream predictFname; // Defaults to empty string (for evaluate()).
      if (writeFiles)
        predictFname << dirPath << wvIndex << "-train_predictions.txt";
      if (!filesystem::exists(predictFname.str())) {
        cout << "Classifying train examples ...\n";
        {
          stringstream identifier;
          identifier << wvIndex << "-Train";
          timer::auto_cpu_timer classifyTrainTimer;
          Utility::evaluate(theta, objective, trainData, identifier.str(),
              predictFname.str());
        }
      }
    }
  }
  
  trainData.clear(); // We no longer need the training data.
  objective->clearKBest(); // We no longer need the k-best lists.

  // Clear the fsts that were cached for the training data.
  if (cachingEnabled) {
    for (size_t i = 0; i < objective->getNumModels(); i++) {
      Model& model = objective->getModel(i);
      model.setCacheEnabled(false);
      model.emptyCache();
    }
  }
  
  if (pipeline && !trainFileSpecified)
    weightVectors.push_back(theta0); // just predict using the loaded weights

  // Load the eval examples from the specified file.
  if (!split && evalFileSpecified && weightVectors.size() > 0) {
    int evalId = 0;
    BOOST_FOREACH(const string& evalFilename, evalFilenames) {
      cout << "Loading " << evalFilename << " ...\n";
      timer::auto_cpu_timer loadEvalTimer;
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
      
      // If we're in pipeline mode and haven't consolidated the alphabets for
      // the various threads, do so now.
      if (pipeline && threadAlphabets.size() > 1) {
        alphabet = objective->combineAlphabets(evalData.getLabelSet());
        alphabet->lock();
        threadAlphabets.clear();
      }
      
      evaluateMultipleWeightVectors(theta0, weightVectors, evalData, objective,
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
