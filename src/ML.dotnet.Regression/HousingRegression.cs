using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Api;

namespace ML.dotnet.Regression.Models
{
    public class HousingRegression
    {
        public static void Run()
        {
            string _trainDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\..\..\datasets", "housing_train_70.csv");
            string _testDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\..\..\datasets", "housing_test_30.csv");
            string resultsPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\..\..\results", "dotnet_results.csv");

            var mlContext = new MLContext();

            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                    {
                        new TextLoader.Column("longitude", DataKind.R4, 0),
                        new TextLoader.Column("latitude", DataKind.R4, 1),
                        new TextLoader.Column("housing_median_age", DataKind.R4, 2),
                        new TextLoader.Column("total_rooms", DataKind.R4, 3),
                        new TextLoader.Column("total_bedrooms", DataKind.R4, 4),
                        new TextLoader.Column("population", DataKind.R4, 5),
                        new TextLoader.Column("households", DataKind.R4, 6),
                        new TextLoader.Column("median_income", DataKind.R4, 7),
                        new TextLoader.Column("median_house_value", DataKind.R4, 8),
                        new TextLoader.Column("<1H OCEAN", DataKind.R4, 9),
                        new TextLoader.Column("INLAND", DataKind.R4, 9),
                        new TextLoader.Column("ISLAND", DataKind.R4, 10),
                        new TextLoader.Column("NEAR BAY", DataKind.R4, 11),
                        new TextLoader.Column("NEAR OCEAN", DataKind.R4, 12),
                    }
            });

            var trainData = reader.Read(_trainDataPath);
            var testData = reader.Read(_testDataPath);

            //Build the training pipeline
            //var pipeline = mlContext.Transforms.Concatenate("Features", "longitude", "latitude", "housing_median_age",
            //        "total_rooms", "total_bedrooms", "population",
            //        "households", "median_income", "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN")
            //    .Append(mlContext.Regression.Trainers.FastTree(label: "median_house_value", features: "Features",
            //        numLeaves: 19, numTrees: 10, minDatapointsInLeafs: 1, learningRate: 0.2D));

            var pipeline = mlContext.Transforms.Concatenate("Features", "longitude", "latitude", "housing_median_age",
                    "total_rooms", "total_bedrooms", "population",
                    "households", "median_income", "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN")
                .Append(mlContext.Regression.Trainers.FastTree(label: "median_house_value", features: "Features"));

            // Train the model
            var model = pipeline.Fit(trainData);
            // Now run the 5-fold cross-validation experiment, using the same pipeline
            var cvResults = mlContext.Regression.CrossValidate(trainData, pipeline, numFolds: 10, labelColumn: "median_house_value");

            var microAccuracies = cvResults.Select(r => r.metrics.RSquared);
            Console.WriteLine($"{microAccuracies.Average()} +-  {GetStandardDeviation(microAccuracies.ToList())}");

            // Compute quality metrics on the test set
            var metrics = mlContext.Regression.Evaluate(model.Transform(testData), label: "median_house_value");

            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for Fast tree          ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn: {metrics.LossFn:0.##}");
            Console.WriteLine($"*       R2 Score: {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.L1:#.##}");
            Console.WriteLine($"*       Squared loss: {metrics.L2:#.##}");
            Console.WriteLine($"*       RMS loss: {metrics.Rms:#.##}");
            Console.WriteLine($"*************************************************");


            /* evaluate model with train data */
            var evaluated = new List<Tuple<float, float>>();
            var predictor = model.MakePredictionFunction<HousingModel, HousingPrediction>(mlContext);
            var testEnumerable = testData.AsEnumerable<HousingModel>(false);

            foreach (var testItem in testEnumerable)
            {
                var prediction = predictor.Predict(testItem);
                evaluated.Add(new Tuple<float, float>(testItem.median_house_value, prediction.Prediction));
            }

            SaveCsv(resultsPath, evaluated);

            Console.WriteLine("Done");
        }

        private static void SaveCsv(string resultsPath, List<Tuple<float, float>> evaluated)
        {
            using (var file = File.CreateText(resultsPath))
            {
                file.WriteLine("dotnet_test,dotnet_pred");

                foreach (var item in evaluated)
                {
                    file.WriteLine($"{item.Item1:#.#},{item.Item2:#.#}");
                }
            }
        }

        private static double GetStandardDeviation(List<double> doubleList)
        {
            var average = doubleList.Average();
            var sumOfDerivation = 0.0;
            foreach (var value in doubleList)
            {
                sumOfDerivation += (value) * (value);
            }
            var sumOfDerivationAverage = sumOfDerivation / (doubleList.Count);

            return Math.Sqrt(sumOfDerivationAverage - (average * average));
        }
    }
}
