using System;
using System.Threading.Tasks;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;

using System.IO;

namespace TestML03
{
    class Program
    {
        public static readonly string DataPath = @"..\..\..\Data\taxi-fare-train.csv";
        public static readonly string TestDataPath = @"..\..\..\Data\taxi-fare-test.csv";
        public static readonly string ModelPath = @"..\..\..\Data\Model.zip";

        static void Main(string[] args)
        {
            Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> modelTask = Train();
            modelTask.Wait();

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> modelResult =
                modelTask.Result;
            Evaluate(modelResult);
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            LearningPipeline pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader<TaxiTrip>(DataPath, useHeader: true, separator: ","));
            pipeline.Add(new ColumnCopier(("fare_amount", "Label")));
            pipeline.Add(new CategoricalOneHotVectorizer("vendor_id",
                                            "rate_code",
                                            "payment_type"));
            pipeline.Add(new ColumnConcatenator("Features",
                                    "vendor_id",
                                    "rate_code",
                                    "passenger_count",
                                    "trip_distance",
                                    "payment_type"));
            pipeline.Add(new FastTreeRegressor());

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            await model.WriteAsync(ModelPath);
            return model;
        }

        static class TestTrips
        {
            internal static readonly TaxiTrip Trip1 = new TaxiTrip
            {
                vendor_id = "VTS",
                rate_code = "1",
                passenger_count = 1,
                trip_distance = 10.33f,
                payment_type = "CHS",
                fare_amount = 0
            };
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var textData = new TextLoader<TaxiTrip>(TestDataPath, useHeader : true, separator : ",");
            var evaluator = new RegressionEvaluator();
            RegressionMetrics matrics = evaluator.Evaluate(model, textData);

            Console.WriteLine("Rms=" + matrics.Rms);
            Console.WriteLine("RSquared=" + matrics.RSquared);

            TaxiTripFarePrediction prediction01 = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predict fare : {0}, actual fare : 29.5", prediction01.fare_amount);
        }
    }
}
