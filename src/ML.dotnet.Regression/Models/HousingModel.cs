using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace ML.dotnet.Regression.Models
{
    public class HousingModel
    {
        [Column("0")]
        public float MedianHomeValue;

        [Column("1")]
        public float CrimesPerCapita;

        [Column("2")]
        public float PercentResidental;

        [Column("3")]
        public float PercentNonRetail;

        [Column("4")]
        public float CharlesRiver;

        [Column("5")]
        public float NitricOxides;

        [Column("6")]
        public float RoomsPerDwelling;

        [Column("7")]
        public float PercentPre40s;

        [Column("8")]
        public float EmploymentDistance;

        [Column("9")]
        public float HighwayDistance;

        [Column("10")]
        public float TaxRate;

        [Column("11")]
        public float TeacherRatio;

        [Column("12")]
        public float BlackIndex;

        [Column("13")]
        public float PercentLowIncome;
    }

    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float Prediction;
    }
}
