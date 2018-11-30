﻿using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.StaticPipe;
using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Learners;

namespace ML.dotnet.Regression.Algorithms
{
   public class FastTree
    {
        public static void FastTreeRegression()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            // this will create a housing.txt file in the filsystem this code will run
            // you can open the file to see the data. 
            string dataFile = Microsoft.ML.SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Creating a data reader, based on the format of the data
            var reader = TextLoader.CreateReader(mlContext, c => (
                        label: c.LoadFloat(0),
                        features: c.LoadFloat(1, 6)
                    ),
                separator: '\t', hasHeader: true);

            // Read the data, and leave 10% out, so we can use them for testing
            var data = reader.Read(dataFile);

            // The predictor that gets produced out of training
            FastTreeRegressionPredictor pred = null;
   
            // Create the estimator
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (r.label, score: mlContext.Regression.Trainers.FastTree(
                                            r.label,
                                            r.features,
                                            numTrees: 100, // try: (int) 20-2000
                                            numLeaves: 20, // try: (int) 2-128
                                            learningRate: 0.2, // try: (float) 0.025-0.4    
                                            minDatapointsInLeafs: 10,
                                        onFit: p => pred = p)
                                )
                        );

            Transformer<(Scalar<float> label, Vector<float> features), (Scalar<float> label, Scalar<float> score), ITransformer> model = learningPipeline.Fit(data);

            var cvResults = mlContext.Regression.CrossValidate(data, learningPipeline, r => r.label, numFolds: 5);
            var averagedMetrics = (
                L1: cvResults.Select(r => r.metrics.L1).Average(),
                L2: cvResults.Select(r => r.metrics.L2).Average(),
                LossFn: cvResults.Select(r => r.metrics.LossFn).Average(),
                Rms: cvResults.Select(r => r.metrics.Rms).Average(),
                RSquared: cvResults.Select(r => r.metrics.RSquared).Average()
            );

            VBuffer<float> weights = default;
            pred.GetFeatureWeights(ref weights);

            var weightsValues = weights.Values;
            Console.WriteLine($"weight 0 - {weightsValues[0]}");
            Console.WriteLine($"weight 1 - {weightsValues[1]}");

            Console.WriteLine($"L1 - {averagedMetrics.L1}");    // 3.091095
            Console.WriteLine($"L2 - {averagedMetrics.L2}");    // 20.351073
            Console.WriteLine($"LossFunction - {averagedMetrics.LossFn}");  // 20.351074
            Console.WriteLine($"RMS - {averagedMetrics.Rms}");              // 4.478358
            Console.WriteLine($"RSquared - {averagedMetrics.RSquared}");    // 0.754977
        }
    }
}