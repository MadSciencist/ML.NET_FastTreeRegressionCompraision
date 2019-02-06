using ML.dotnet.Regression.Models;
using System;

namespace ML.dotnet.Regression
{
    internal class Program
    {
        static void Main(string[] args)
        {
            HousingRegression.Run();
            Console.Read();
        }
    }
}