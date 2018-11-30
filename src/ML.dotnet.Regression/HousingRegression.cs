using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers.FastTree;

namespace ML.dotnet.Regression.Models
{
    public class HousingRegression
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\..\..\datasets", "housing.txt");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\..\..\datasets", "housing-test.txt");
        static TextLoader _textLoader;

        public static void Run()
        {
            var mlContext = new MLContext();

            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
                {
                    Separator = "\t",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("MedianHomeValue", DataKind.R4, 0),
                        new TextLoader.Column("CrimesPerCapita", DataKind.R4, 1),
                        new TextLoader.Column("PercentResidental", DataKind.R4, 2),
                        new TextLoader.Column("PercentNonRetail", DataKind.R4, 3),
                        new TextLoader.Column("CharlesRiver", DataKind.R4, 4),
                        new TextLoader.Column("NitricOxides", DataKind.R4, 5),
                        new TextLoader.Column("RoomsPerDwelling", DataKind.R4, 6),
                        new TextLoader.Column("PercentPre40s", DataKind.R4, 7),
                        new TextLoader.Column("EmploymentDistance", DataKind.R4, 8),
                        new TextLoader.Column("HighwayDistance", DataKind.R4, 9),
                        new TextLoader.Column("TaxRate", DataKind.R4, 10),
                        new TextLoader.Column("TeacherRatio", DataKind.R4, 11),
                        new TextLoader.Column("BlackIndex", DataKind.R4, 12),
                        new TextLoader.Column("PercentLowIncome", DataKind.R4, 13),
                    }
                }
            );

            var model = Train(mlContext, _trainDataPath);
            Evaluate(mlContext, model);
            TestSinglePrediction(mlContext, model);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.CopyColumns("MedianHomeValue", "Label")
                .Append(mlContext.Transforms.Concatenate("Features", "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                        "PercentPre40s", "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio", "BlackIndex", "PercentLowIncome"))
                .Append(mlContext.Regression.Trainers.FastTree(numLeaves: 10, numTrees: 20, minDatapointsInLeafs: 5, learningRate: 0.2));

            Console.WriteLine("=============== Create and Train the Model ===============");

            var model = pipeline.Fit(dataView);

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            var cvResults = mlContext.Regression.CrossValidate(dataView, pipeline, 2);
            var averagedMetrics = (
                L1: cvResults.Select(r => r.metrics.L1).Average(),
                L2: cvResults.Select(r => r.metrics.L2).Average(),
                LossFn: cvResults.Select(r => r.metrics.LossFn).Average(),
                Rms: cvResults.Select(r => r.metrics.Rms).Average(),
                RSquared: cvResults.Select(r => r.metrics.RSquared).Average()
            );
            Console.WriteLine(averagedMetrics);

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);

            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            Console.WriteLine($"*************************************************");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            //Prediction test
            // Create prediction function and make prediction.
            var predictionFunction = model.MakePredictionFunction<HousingModel, HousingPrediction>(mlContext);

            var hosuingSample = new HousingModel()
            {
                MedianHomeValue = 0f, //13.80
                CrimesPerCapita = 8.64476f,
                PercentResidental = 0f,
                PercentNonRetail =  18.1f,
                CharlesRiver = 0f,
                NitricOxides = 0.693f,
                RoomsPerDwelling = 6.1930f,
                PercentPre40s = 92.6f,
                EmploymentDistance = 1.7912f,
                HighwayDistance = 24f,
                TaxRate = 666f,
                TeacherRatio = 20.20f,
                BlackIndex = 396.9f,
                PercentLowIncome = 15.17f
            };

            var prediction = predictionFunction.Predict(hosuingSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted: {prediction.Prediction:0.####}, actual: 13.80");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
