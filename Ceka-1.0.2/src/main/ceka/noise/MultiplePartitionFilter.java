package ceka.noise;

import java.util.ArrayList;

import weka.classifiers.Classifier;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.utils.DatasetManipulator;

/**
 * Article:<br>
 * Khoshgoftaar, T.M., & Rebours, P. (2007). Improving software quality prediction 
 * by noise filtering techniques. J. Comput. Sci. Technol. 22, pp. 387-396.
 *
 */

public class MultiplePartitionFilter extends Filter
{
	private int filteringLevel = 3;
	private int nFold = 5;
	
	public MultiplePartitionFilter(int filteringLevel,int nFold)
	{
		this.filteringLevel = filteringLevel;
		this.nFold = nFold;
	}
	
	public void filterNoise(Dataset dataset, Classifier[] classifiers) throws Exception
	{
		int nClassifiers = classifiers.length;
		System.out.println("Entering function");
		createInternal(dataset);
		ArrayList<ArrayList<Double>> predictors = new ArrayList<ArrayList<Double>>();
		Dataset[] datasets = DatasetManipulator.split(dataset, nFold, true);
		for(Dataset d : datasets)
		{
			for(Classifier c : classifiers)
			{
				predictors.add(new ArrayList<Double>());
				c.buildClassifier(d);
				for(int e = 0; e < dataset.getExampleSize(); e++)
				{
					double predict = c.classifyInstance(dataset.getExampleByIndex(e));
					predictors.get(predictors.size() - 1).add(predict);
				}
			}
		}
		
		int numberIncorrect = 0;
		
		for(int i = 0; i < predictors.get(0).size(); i++)
		{
			numberIncorrect = 0;
			Example thisExample = dataset.getExampleByIndex(i);
			double correctLabel = thisExample.value(thisExample.numAttributes() - 1);
			for(int f = 0; f < nClassifiers * nFold; f++)
		    {
				double predict = predictors.get(f).get(i);
				if(predict != (int)correctLabel)
					numberIncorrect++;
			}
			
			if(numberIncorrect >= filteringLevel)
			{
				noiseDataset.addExample(thisExample);
			}
			else
			{
				cleanedDataset.addExample(thisExample);	
			}
		}
	}
}