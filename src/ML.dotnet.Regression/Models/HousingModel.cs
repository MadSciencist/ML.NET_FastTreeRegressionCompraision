using Microsoft.ML.Runtime.Api;

namespace ML.dotnet.Regression.Models
{
    public class HousingModel
    {
        [Column("0", name: "longitude")]
        public float longitude;

        [Column("1", name: "latitude")]
        public float latitude;

        [Column("2", name: "housing_median_age")]
        public float housing_median_age;

        [Column("3", name: "total_rooms")]
        public float total_rooms;

        [Column("4", name: "total_bedrooms")]
        public float total_bedrooms;

        [Column("5", name: "population")]
        public float population;

        [Column("6", name: "households")]
        public float households;

        [Column("7", name: "median_income")]
        public float median_income;

        [Column("8", name: "median_house_value")]
        public float median_house_value;

        [Column("9", name: "<1H OCEAN")]
        public float Ocean1h;

        [Column("10", name: "INLAND")]
        public float INLAND;

        [Column("11", name: "ISLAND")]
        public float ISLAND;

        [Column("12", name: "NEAR BAY")]
        public float NearBay;

        [Column("13", name: "NEAR OCEAN")]
        public float NearOceam;
    }

    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float Prediction;
    }
}
