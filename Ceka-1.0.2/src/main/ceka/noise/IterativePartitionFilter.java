package ceka.noise;

import ceka.core.Category;
import weka.classifiers.Classifier;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.utils.DatasetManipulator;
import java.util.ArrayList;

/**
 * Article:<br>
 * Khoshgoftaar, T.M., & Rebours, P. (2007). Improving software quality prediction 
 * by noise filtering techniques. J. Comput. Sci. Technol. 22, pp. 387-396.
 *
 */
public class IterativePartitionFilter extends Filter {

        public IterativePartitionFilter(int nFold, String votingScheme, double percentage) {
            this.nFold = nFold;
            this.votingScheme = votingScheme;
            this.percentage = percentage;
        }
        
	@Override
	public void filterNoise(Dataset dataset, Classifier [] classifier) throws Exception {
            createInternal(dataset);
            ArrayList<Double> percentages = new ArrayList();
            int size = dataset.getExampleSize();
            boolean stop = false;
            
            while (!stop) 
            {
                ArrayList<ArrayList<Double>> predictors = new ArrayList();
                Dataset [] datasets = DatasetManipulator.split(dataset, nFold, true);
                for (int i = 0; i < nFold; i++) 
                {
                    predictors.add(new ArrayList<Double>());
                    Dataset thisFold = datasets[i];
                    // train a model on each fold of the training set
                    classifier[0].buildClassifier(thisFold);
                    for(int j = 0; j < dataset.getExampleSize(); j++)
                    {
                        double predict = classifier[0].classifyInstance(dataset.getExampleByIndex(j));
                        predictors.get(i).add(predict);
                    }
                }
                int noisyExamplesFound = 0;
                Dataset tempDataset = new Dataset(dataset,0);
                int numCateSize = dataset.getCategorySize();
		for (int i = 0; i < numCateSize; i++) 
                {
			Category cate = dataset.getCategory(i);
			tempDataset.addCategory(cate.copy());
                }
                for(int e = 0; e < dataset.getExampleSize(); e++)
                {
                    Example thisExample = dataset.getExampleByIndex(e);
                    double percentCorrect = 0.0;
                    double correctLabel = thisExample.value(thisExample.numAttributes() - 1);
                    for(int n = 0; n < nFold; n++)
                    {
                        if(predictors.get(n).get(e).intValue() == (int)correctLabel) 
                        {
                            percentCorrect++;
                        }
                    }
                    percentCorrect /= (double)nFold;
                    if(votingScheme.equals("consensus"))
                    {
                        if(percentCorrect == 0.0)
                        {
                            noiseDataset.addExample(thisExample);
                            noisyExamplesFound++;
                        }
                        else
                            tempDataset.addExample(thisExample);
                    }
                    else
                    {
                        if(percentCorrect < .5)
                        {
                            noiseDataset.addExample(thisExample);
                            noisyExamplesFound++;
                        }
                        else
                        {
                            tempDataset.addExample(thisExample);
                        }
                    }
                }
                dataset = tempDataset;
                
                percentages.add(new Double((double)noisyExamplesFound/(double)size));
                if(percentages.size() >= 3)
                {
                    if(percentages.get(percentages.size() - 1) < percentage
                        && percentages.get(percentages.size() - 2) < percentage
                        && percentages.get(percentages.size() - 3) < percentage)
                        stop = true;
                }
            }
            super.cleanedDataset = dataset;
	}
        
        private int nFold = 5;
        private String votingScheme = "majority";
        private double percentage = 0.01;
}