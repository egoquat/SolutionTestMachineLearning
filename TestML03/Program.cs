using System;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;

namespace TestML03
{
    class Program
    {
        public static readonly string DataPath = @"..\..\..\Data\taxi-fare-train.csv";
        public static readonly string TestDataPath = @"..\..\..\Data\taxi-fare-test.csv";
        public static readonly string ModelPath = @"..\..\..\Models\Model.zip";

        static void Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = Train();

        }

        public static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<TaxiTrip>(DataPath, useHeader: true, separator: ","),
                new ColumnCopier(("fare_amount", "Label")),
                new CategoricalOneHotVectorizer("vendor_id",
                                             "rate_code",
                                             "payment_type"),
                new ColumnConcatenator("Features",
                                                "vendor_id",
                                                "rate_code",
                                                "passenger_count",
                                                "trip_distance",
                                                "payment_type"),
                new FastTreeRegressor()
            };

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
        }
    }
}
