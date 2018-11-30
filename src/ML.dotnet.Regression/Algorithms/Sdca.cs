using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.StaticPipe;
using System;

namespace ML.dotnet.Regression.Algorithms
{
    public class Sdca
    {
        public static void ProcessSdca()
        {
            string dataFile = DatasetUtils.DownloadHousingRegressionDataset();

            var mlContext = new MLContext();

            // Creating a data reader, based on the format of the data
            var reader = TextLoader.CreateReader(mlContext, c => (
                    label: c.LoadFloat(0),
                    features: c.LoadFloat(1, 6)
                ),
                separator: '\t', hasHeader: true);

            var data = reader.Read(dataFile);
            (var trainData, var testData) = mlContext.Regression.TrainTestSplit(data, testFraction: 0.1);

            LinearRegressionPredictor pred = null;

            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (r.label, score: mlContext.Regression.Trainers.Sdca(
                        r.label,
                        r.features,
                        l1Threshold: 0f,
                        maxIterations: 100,
                        onFit: p => pred = p)
                    )
                );


            // Fit this pipeline to the training data
            var model = learningPipeline.Fit(trainData);

            // Check the weights that the model learned
            VBuffer<float> weights = default;
            pred.GetFeatureWeights(ref weights);

            var weightsValues = weights.Values;
            Console.WriteLine($"weight 0 - {weightsValues[0]}");
            Console.WriteLine($"weight 1 - {weightsValues[1]}");

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions, r => r.label, r => r.score);

            Console.WriteLine($"L1 - {metrics.L1}");  // 3.7226085
            Console.WriteLine($"L2 - {metrics.L2}");  // 24.250636
            Console.WriteLine($"LossFunction - {metrics.LossFn}");  // 24.25063
            Console.WriteLine($"RMS - {metrics.Rms}");  // 4.924493
            Console.WriteLine($"RSquared - {metrics.RSquared}");  // 0.565467
        }
    }
}
