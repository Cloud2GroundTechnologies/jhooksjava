package org.dataalgorithms.machinelearning.naivebayes.diabetes;

import scala.Tuple2;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.classification.NaiveBayesModel;

/**
 * 
 *
 * @author John Hooks code tested by Haibo Chen
 *
 */
public class PredictDiabetes {

    private static final Logger THE_LOGGER = Logger.getLogger(PredictDiabetes.class);


    public static void main(String[] args) throws Exception {
        Util.printArguments(args);
        if (args.length != 2) {
            throw new RuntimeException("usage: PredictDiabetes <query-data-path> <saved-model-path> ");
        }

        //
        String queryDataPath = args[0];
        String savedModelPath = args[1];
        THE_LOGGER.info("--- queryDataPath=" + queryDataPath);
        THE_LOGGER.info("--- savedModelPath=" + savedModelPath);

        // create a Factory context object
        JavaSparkContext context = Util.createJavaSparkContext("PredictDiabetes");

        //
        // create query data set
        // input records format: <feature-1><,>...<,><feature-8>
        //
        JavaRDD<String> queryRDD = context.textFile(queryDataPath);
        JavaRDD<Vector> query = Util.createFeatureVector(queryRDD);

        //
        // load the built model from the saved path
        //
        final NaiveBayesModel model = NaiveBayesModel.load(context.sc(), savedModelPath);

        //
        // predict the query data
        // JavaPairRDD<Vector, Double> = JavaPairRDD<Vector as input, Double prediction as output>
        //
        JavaPairRDD<Vector, Double> predictionAndLabel
                = query.mapToPair(new PairFunction<Vector, Vector, Double>() {
                    @Override
                    public Tuple2<Vector, Double> call(Vector v) {
                        // predict values for a single data point using the model trained.
                        double prediction = model.predict(v);
                        return new Tuple2<Vector, Double>(v, prediction);
                    }
        });
        
        //
        // DEBUG/VIEW predictions:
        //
        Iterable<Tuple2<Vector, Double>> predictions = predictionAndLabel.collect();
        for (Tuple2<Vector, Double> p : predictions) {
            THE_LOGGER.info("input: "+ p._1);
            THE_LOGGER.info("prediction: "+ p._2);
        }

        // done
        context.close();
    }

}
