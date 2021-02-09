
package ceka.noise;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.utils.DatasetManipulator;
import weka.classifiers.Classifier;

/**
 * A noise filter modeled after the Majority Vote Ensemble. <br><br>
 * Article: <br>
 * Brodley and Friedl, "Identifying Mislabeled Training Data," 
 * Journal of Artificial Intelligence Research 11 (1999) 131-167.
 */
public class MajorityFilter extends Filter
{
    private int numFolds;
    public MajorityFilter(int numFolds)
    {
        this.numFolds = numFolds;
    }
    
    @Override
    public void filterNoise(Dataset dataset, Classifier[] classifiers) throws Exception
    {
        createInternal(dataset);
        Dataset[] folds = DatasetManipulator.split(dataset, numFolds, true);
        int numClassifiers = classifiers.length;
        double[][] classifications = new double[numClassifiers][dataset.getExampleSize()];
        
        for(int c = 0; c < numClassifiers; c++)
        {
            int exampleIndex = 0;
            for(int i = 0; i < numFolds; i++)
            {
                Dataset[] dividedDatasets = DatasetManipulator.pickCombine(folds, i);
                Dataset thisDataset = dividedDatasets[1];
                Dataset trainDataset = dividedDatasets[0];
                classifiers[c].buildClassifier(trainDataset);
                for(int j = 0; j < thisDataset.getExampleSize(); j++)
                {
                    Example e = thisDataset.getExampleByIndex(j);
                    classifications[c][exampleIndex] = classifiers[c].classifyInstance(e);
                    exampleIndex++;
                    //System.out.println("Classifier " + c + " Fold " + i + " Example " + e.getId());
                }
            }
        }
        
        for(int i = 0; i < dataset.getExampleSize(); i++)
        {
            Example e = dataset.getExampleByIndex(i);
            int numCorrect = 0;
            for(int c = 0; c < numClassifiers; c++)
            {
                if((int)classifications[c][i] == e.getTrainingLabel())
                    numCorrect++;
            }
            if((double)numCorrect / (double)numClassifiers > .5)
                cleanedDataset.addExample(e);
            else
                noiseDataset.addExample(e);
        }
    }
}
