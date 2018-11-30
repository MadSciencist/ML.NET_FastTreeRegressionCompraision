using ML.dotnet.Regression.Algorithms;
using System;
using ML.dotnet.Regression.Models;

namespace ML.dotnet.Regression
{
    class Program
    {
        static void Main(string[] args)
        {
            //TaxiFareRegression.Run();

            FastTree.FastTreeRegression();

            HousingRegression.Run();
            Console.Read();

        }
    }
}