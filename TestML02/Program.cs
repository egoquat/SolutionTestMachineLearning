using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace TestML02
{
    class Program
    {
        public class SentimentData
        {
            [Column(ordinal: "0")]
            public string SentimentText;
            [Column(ordinal: "1", name: "Label")]
            public float Sentiment;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Sentiment = false;
        }

        const string _dataPath = @"..\..\..\data\sentiment labelled sentences\imdb_labelled.txt";
        const string _testDataPath = @"..\..\..\data\sentiment labelled sentences\yelp_labelled.txt";

        public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict()
        {
            string pathCurrent = System.IO.Directory.GetCurrentDirectory();

            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<SentimentData>(_dataPath, useHeader: false, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            PredictionModel<SentimentData, SentimentPrediction> model = null;
            try
            {
                model = pipeline.Train<SentimentData, SentimentPrediction>();

                IEnumerable<SentimentData> sentiments = new[]
                {
                    new SentimentData
                    {
                        SentimentText = "Contoso's 11 is a wonderful experience",
                        Sentiment = 0
                    },
                    new SentimentData
                    {
                        SentimentText = "The acting in this movie is very bad",
                        Sentiment = 0
                    },
                    new SentimentData
                    {
                        SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
                        Sentiment = 0
                    },
                    new SentimentData
                    {
                        SentimentText = "horrible.",
                        Sentiment = 0
                    },
                    new SentimentData
                    {
                        SentimentText = "oh my godness. wonderful.",
                        Sentiment = 0
                    },
                    new SentimentData
                    {
                        SentimentText = "Contoso's 11 is not an ordinary expirence. just fine.",
                        Sentiment = 0
                    },
                    new SentimentData
                    {
                        SentimentText = "It's based on an personal condition.",
                        Sentiment = 0
                    }
                };

                IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
                var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

                Console.WriteLine();
                Console.WriteLine("Result Sentiment Predictions >>>>>>>>>>>>>>>>>>");
                Console.WriteLine("---------------------");
                foreach ((SentimentData, SentimentPrediction) item in sentimentsAndPredictions)
                {
                    SentimentData sentiment = item.Item1;
                    SentimentPrediction prediction = item.Item2;
                    Console.WriteLine($"Sentiment: {sentiment.SentimentText} | Prediction: {(prediction.Sentiment ? "Positive" : "Negative")}");
                }
                Console.WriteLine();
            }
            catch(Exception e)
            {
                Console.WriteLine("error:" + e.Message);
            }

            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader<SentimentData>(_testDataPath, useHeader: false, separator: "tab");
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        static void Main(string[] args)
        {
            var model = TrainAndPredict();

            Evaluate(model);
        }
    }
}
