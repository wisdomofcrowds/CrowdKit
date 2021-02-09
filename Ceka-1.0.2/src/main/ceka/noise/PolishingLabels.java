/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ceka.noise;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.utils.DatasetManipulator;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

/**
 *
 * @author jian
 */
public class PolishingLabels 
{
    private Classifier classifier;

    public PolishingLabels(Classifier classifier)
    {
        this.classifier = classifier;
    }
    
    public Dataset polishLabels(Dataset dataset) throws Exception
    {
    	/* rewrite by Jing Zhang */
        //Dataset polishedDataset = dataset.makeCopy();
    	Dataset polishedDataset = dataset.generateEmpty();
    	for (int i =0; i < dataset.getExampleSize(); i++) 
    	{
    		polishedDataset.addExample(dataset.getExampleByIndex(i));
    	}
    	for(int i = 0; i < dataset.getCategorySize(); i++)
    	{
    		polishedDataset.addCategory(dataset.getCategory(i));
    	}
    	/*end*/
        
    	int[][] votes = new int[polishedDataset.getExampleSize()][10];
        Classifier[] classifiers = new Classifier[10];
        Dataset[] folds = DatasetManipulator.split(polishedDataset, 10, false);
        for(int i = 0; i < 10; i++)
        {
            classifiers[i] = new J48();
            classifiers[i].buildClassifier(folds[i]);
        }
        
        int exampleIndex = 0;
        for(int i = 0; i < 10; i++)
        {
            for(int j = 0; j < polishedDataset.getExampleSize(); j++)
            {
                Example e = polishedDataset.getExampleByIndex(j);
                votes[j][i] = (int)classifiers[i].classifyInstance(e);
            }
        }

        for(int i = 0; i < polishedDataset.getExampleSize(); i++)
        {
            Example e = polishedDataset.getExampleByIndex(i);
            int[] counts = new int[polishedDataset.getCategorySize()];
            for(int j = 0; j < classifiers.length; j++)
            {
                counts[votes[i][j]]++;
            }
            double max = Double.NEGATIVE_INFINITY;
            ArrayList<Integer> maxIndices = new ArrayList();
            for(int j = 0; j < counts.length; j++)
            {
                if(counts[j] > max)
                {
                    max = counts[j];
                    maxIndices.clear();
                    maxIndices.add(j);
                }
                else if(counts[j] == max)
                {
                    maxIndices.add(j);
                }
            }
            if(maxIndices.size() == 1)
            {
                e.getIntegratedLabel().setValue(maxIndices.get(0));
                e.setTrainingLabel(maxIndices.get(0)); 
            }
            else
            {
                boolean changeLabel = true;
                for(int k = 0; k < maxIndices.size(); k++)
                {
                    if(e.getIntegratedLabel().getValue() == maxIndices.get(k))
                    {
                        changeLabel = false;
                        break;
                    }
                }
                if(changeLabel)
                {
                    Random rand = new Random();
                    int index = rand.nextInt(maxIndices.size());
                    e.getIntegratedLabel().setValue(maxIndices.get(index));
                    e.setTrainingLabel(maxIndices.get(index)); 
                }
            }
        }

        return polishedDataset;
    }
}
