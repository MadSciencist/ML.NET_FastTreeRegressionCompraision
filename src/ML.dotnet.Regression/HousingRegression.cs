using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using System;
using System.IO;
using System.Linq;

namespace ML.dotnet.Regression.Models
{
    public class HousingRegression
    {
        public static void Run()
        {
            string _trainDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\..\..\datasets", "housing_train_70.csv");
            string _testDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\..\..\datasets", "housing_test_30.csv");

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
                        new TextLoader.Column("<1H OCEAN,INLAND", DataKind.R4, 9),
                        new TextLoader.Column("ISLAND", DataKind.R4, 10),
                        new TextLoader.Column("NEAR BAY", DataKind.R4, 11),
                        new TextLoader.Column("NEAR OCEAN", DataKind.R4, 12),
                    }
            });

            var trainData = reader.Read(_trainDataPath);
            var testData = reader.Read(_testDataPath);

            // Build the training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "longitude", "latitude", "housing_median_age",
                    "total_rooms", "total_bedrooms", "population",
                    "households", "median_income", "<1H OCEAN,INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN")
                .Append(mlContext.Regression.Trainers.FastTree(label: "median_house_value", features: "Features"));

            // Train the model
            var model = pipeline.Fit(trainData);

            // Now run the 5-fold cross-validation experiment, using the same pipeline
            var cvResults = mlContext.Regression.CrossValidate(trainData, pipeline, numFolds: 10, labelColumn: "median_house_value");

            var microAccuracies = cvResults.Select(r => r.metrics.RSquared);
            Console.WriteLine(microAccuracies.Average());

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
        }
    }
}
