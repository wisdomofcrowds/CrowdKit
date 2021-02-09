package ceka.noise;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import ceka.converters.FileLoader;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.noise.SelfTrainCorrection.ConfIndex;
import ceka.utils.DescendingElement;

/**
 * Uses the paradigm of using labeled examples in a dataset to label the most
 * confident unlabeled examples, and applies it to re-labeling examples
 * found to be noisy by a previous filtering algorithm. <br> <br>
 * Article: <br>
 * Isaac Triguero, Jose A Saez, Julian Luengo, Salvador Garca, and Francisco Herrera. On
the characterization of noise filters for self-training semi-supervised in nearest neighbor
classification. Neurocomputing, 132:30–41, 2014.
 * @author Bryce
 */
public class SelfTrainCorrection 
{
	private int MAXITER = 0;
	private Dataset cleanExamples = null;
	private Dataset noisyExamples = null;
        private int numberOfClasses = 0;
        private double[] prototypesPerClass = null;

        /**
         * Constructor. Instantiates the various parameters needed to execute
         * the algorithm given the clean and noisy datasets, and the proportion
         * of noisy examples the user desires to re-label.
         * @param cleanExamples
         * @param noisyExamples
         * @param proportion 
         */
	public SelfTrainCorrection(Dataset cleanExamples, Dataset noisyExamples, double proportion)
	{
            numberOfClasses = cleanExamples.getCategorySize();
            prototypesPerClass = new double[numberOfClasses];
            double minimum = Double.POSITIVE_INFINITY;
            for(int i = 0; i < numberOfClasses; i++)
            {
                    int numPrototypes = 0;
                    for(int j = 0; j < cleanExamples.getExampleSize(); j++)
                    {
                            Example e = cleanExamples.getExampleByIndex(j);
                            int elabel = e.getTrainingLabel();
                            int cate = cleanExamples.getCategory(i).getValue();
                            if(elabel == cate) numPrototypes++;
                    }
                    prototypesPerClass[i] = (double)numPrototypes / (double)cleanExamples.getExampleSize();
                    if(prototypesPerClass[i] < minimum)
                            minimum = prototypesPerClass[i];
            }
            for(int j = 0; j < numberOfClasses; j++)
            {
                    prototypesPerClass[j] = Math.round(prototypesPerClass[j] / minimum);
            }
            
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
            int examplesAddedPerIteration = 0;
            for(int i = 0; i < prototypesPerClass.length; i++)
            	examplesAddedPerIteration += prototypesPerClass[i];
            MAXITER = (int)Math.round((double)noisyExamples.getExampleSize() / (double)examplesAddedPerIteration * proportion);	
	}

	/**
         * Will use the clean examples to re-classify the noisy examples, which were determined prior
         * to this function's execution.
         * */
	public Dataset[] correction(Classifier classifier) throws Exception
	{
		int z = 0;
		
		while(z < MAXITER && noisyExamples.getExampleSize() > 0)
		{
			//System.out.println("MAXITER = " + MAXITER + " Noise Size=" + noisyExamples.getExampleSize());
			double[][] classificationConfidences = new double[noisyExamples.getExampleSize()][numberOfClasses];
			//NaiveBayes nb = new NaiveBayes();
			classifier.buildClassifier(cleanExamples);
			for(int i = 0; i < noisyExamples.getExampleSize(); i++)
			{
				Example noise = noisyExamples.getExampleByIndex(i);
				classifier.classifyInstance(noise);
				classificationConfidences[i] = classifier.distributionForInstance(noise);
			}

			ArrayList<ConfIndex> confIndexes = new ArrayList<ConfIndex>();
			for(int i = 0; i < noisyExamples.getExampleSize(); i++)
			{
				confIndexes.add(new ConfIndex(maxElement(classificationConfidences[i]),indexOfMaxElement(classificationConfidences[i])));
			}
			Collections.sort(confIndexes);
			for(int a = 0; a < prototypesPerClass.length; a++)
			{
				int examplesAdded = 0;
				int iteration = 0;
				while(examplesAdded < prototypesPerClass[a] && iteration < noisyExamples.getExampleSize())
				{
					for(int j = 0; j < noisyExamples.getExampleSize(); j++)
					{
						Example e = noisyExamples.getExampleByIndex(j);
						if(maxElement(classificationConfidences[j]) == confIndexes.get(iteration).getConf() && 
								cleanExamples.getCategory(a).getValue() == indexOfMaxElement(classificationConfidences[j]))
						{
							e.getIntegratedLabel().setValue(a);
                            e.setTrainingLabel(a);
							cleanExamples.addExample(e);
							noisyExamples.simpleRemoveExampleByIndex(j);				
							examplesAdded++;
							break;
						}
					}
					iteration++;
				}
			}
			z++;
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
        
        class ConfIndex implements Comparable<ConfIndex>
{
	private double conf;
	private int index;
	
	public ConfIndex(double d, int i)
	{
		conf = d;
		index = i;
	}
	
	public int getIndex() {return index;}
	public double getConf() {return conf;}
	
	public int compareTo(ConfIndex other)
	{
		if(this.conf < other.conf) return -1;
		else if(this.conf > other.conf) return 1;
		else return 0;
	}
}
}