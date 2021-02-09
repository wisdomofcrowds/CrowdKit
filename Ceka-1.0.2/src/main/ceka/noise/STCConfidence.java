package ceka.noise;

import weka.classifiers.Classifier;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;

/**
 * Uses the paradigm of using labeled examples in a dataset to label the most
 * confident unlabeled examples, and applies it to re-labeling examples
 * found to be noisy by a previous filtering algorithm. <br>
 * <br>
 * Article: <br>
 * Isaac Triguero, Jose A Saez, Julian Luengo, Salvador Garca, and Francisco Herrera. On
the characterization of noise filters for self-training semi-supervised in nearest neighbor
classification. Neurocomputing, 132:30–41, 2014.
 * @author Bryce
 * @see 
*  @version the difference between this version and {@link SelfTrainCorrect} is that in this version
*  the user should specify a confidence level which indicates whether an instance is a noise.
 */
public class STCConfidence 
{
	private double confidenceThreshold = 0;
	private Dataset cleanExamples = null;
	private Dataset noisyExamples = null;
    /**
     * Constructor. Instantiates the various parameters needed to execute
     * the algorithm given the clean and noisy datasets, and the proportion
     * of noisy examples the user desires to re-label.
     * @param cleanExamples
     * @param noisyExamples
     * @param confidence level 
     */
	public STCConfidence(Dataset cleanExamples, Dataset noisyExamples, double threshold)
	{
            cleanExamples.getCategorySize();
            this.confidenceThreshold = threshold;
            this.cleanExamples = cleanExamples.generateEmpty();
            int numCateSize = cleanExamples.getCategorySize();
            for (int i = 0; i < numCateSize; i++) 
            {
                    Category cate = cleanExamples.getCategory(i);
                    this.cleanExamples.addCategory(cate.copy());
            }
            for(int i = 0; i < cleanExamples.getExampleSize(); i++)
            {
                this.cleanExamples.addExample(cleanExamples.getExampleByIndex(i));
            }
            this.noisyExamples = noisyExamples.generateEmpty();
            numCateSize = noisyExamples.getCategorySize();
            for (int i = 0; i < numCateSize; i++) 
            {
                    Category cate = noisyExamples.getCategory(i);
                    this.noisyExamples.addCategory(cate.copy());
            }
            for(int i = 0; i < noisyExamples.getExampleSize(); i++)
            {
                this.noisyExamples.addExample(noisyExamples.getExampleByIndex(i));
            }
        }

	/**
         * Will use the clean examples to re-classify the noisy examples, which were determined prior
         * to this function's execution.
         * */
	public Dataset[] correction(Classifier classifier) throws Exception
	{
                boolean stop = false;
		while(!stop)
		{
                        stop = true;
			classifier.buildClassifier(cleanExamples);
			for(int i = noisyExamples.getExampleSize() - 1; i >= 0; i--)
			{
				Example noise = noisyExamples.getExampleByIndex(i);
				classifier.classifyInstance(noise);
                double[] dist = classifier.distributionForInstance(noise);
                if(maxElement(dist) >= confidenceThreshold)
                {
                    stop = false;
                    int label = indexOfMaxElement(dist);
                    noise.getIntegratedLabel().setValue(label);
                    noise.setTrainingLabel(label);
                    cleanExamples.addExample(noise);
                    noisyExamples.simpleRemoveExampleByIndex(i);
                }
			}
		}
		
		Dataset[] cleanAndNoisyDatasets = new Dataset[2];
		cleanAndNoisyDatasets[0] = cleanExamples;
		cleanAndNoisyDatasets[1] = noisyExamples;
		return cleanAndNoisyDatasets;
	}
	
	private double maxElement(double[] doubles)
	{
		double max = Double.NEGATIVE_INFINITY;
		for(double d : doubles)
		{
			if(d > max)
				max = d;
		}
		return max;
	}
	
	private int indexOfMaxElement(double[] doubles)
	{
		double max = Double.NEGATIVE_INFINITY;
		int maxIndex = -1;
		for(int i = 0; i < doubles.length; i++)
		{
			if(doubles[i] > max)
			{
				max = doubles[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
}