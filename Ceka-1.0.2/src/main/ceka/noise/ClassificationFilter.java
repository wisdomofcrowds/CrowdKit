package ceka.noise;

import weka.classifiers.Classifier;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.utils.DatasetManipulator;

/**
 * 
 * Article: <br>
 * Gamberger, D., Boskovic, R., Lavrac, N., & Groselj, C. (1999). Experiments with
 * noise filtering in a medical domain. In: Proc. the 16th ICML, pp. 143-151.
 *
 */
public class ClassificationFilter extends Filter {

	public ClassificationFilter (int nFold){
		this.nFold = nFold;
	}
	
	@Override
	public void filterNoise(Dataset dataset, Classifier[] classifier) throws Exception {
		createInternal(dataset);
		Dataset [] datasets = DatasetManipulator.split(dataset, nFold, true);
		for (int i = 0; i < nFold; i++) {
			Dataset [] trainAndTest = DatasetManipulator.pickCombine(datasets, i);
			Dataset trainDataset = trainAndTest[0];
			Dataset testDataset = trainAndTest[1];
			// train a model
			classifier[0].buildClassifier(trainDataset);
			// test one by one
			for (int j = 0; j < testDataset.getExampleSize(); j++) {
				Example testExample = testDataset.getExampleByIndex(j);
				double predict = classifier[0].classifyInstance(testExample);
				if ((int)predict != (int) testExample.value(testExample.numAttributes() - 1))
					noiseDataset.addExample(testExample);
				else
					cleanedDataset.addExample(testExample);
			}
		}
	}	
		
	private int nFold = 5;
	
}
