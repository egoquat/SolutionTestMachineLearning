using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;

namespace TestML03
{
    class TaxiTripFarePrediction
    {
        float fare_amount;
    }

    class TaxiTrip
    {
        LearningPipeline Pipeline = new LearningPipeline
        {
            new TextLoader<TaxiTrip>(Program.DataPath, useHeader: true, separator: ","),
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
    }
}
