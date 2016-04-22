package org.dataalgorithms.machinelearning.naivebayes.diabetes;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;


/**
 * 
 * @author John Hooks code tested by Haibo Chen
 * 
 */
public class BuildDiabetesModel {
    
    private static final Logger THE_LOGGER = Logger.getLogger(BuildDiabetesModel.class);
            

    public static void main(String[] args) throws Exception {
        Util.printArguments(args);
        if (args.length != 2) {
            throw new RuntimeException("usage: BuildDiabetesModel <training-data-path> <saved-path-for-built-model>");
        }

        //
        String trainingPath = args[0];
        THE_LOGGER.info("--- trainingPath=" + trainingPath);
        //
        String savedPathForBuiltModel = args[1];
        THE_LOGGER.info("--- savedPathForBuiltModel=" + savedPathForBuiltModel);

        // create a Factory context object
        JavaSparkContext context = Util.createJavaSparkContext("BuildDiabetesModel");
        

        //
        // create training data set
        // input records format: <feature-1><,>...<,><feature-8><,><classification>
        //
        JavaRDD<String> trainingRDD = context.textFile(trainingPath);        
        JavaRDD<LabeledPoint> training  = Util.createLabeledPointRDD(trainingRDD);
        
        //
        // create a model from the given training data set
        //
        final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        
        //
        // save the built model for future use
        //
        //      public void save(SparkContext sc, java.lang.String path)
        //      Description; Save this model to the given path.       
        model.save(context.sc(), savedPathForBuiltModel);
        
        
        // done
        context.close();
    }
    
}

