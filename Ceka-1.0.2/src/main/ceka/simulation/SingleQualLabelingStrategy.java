/**
 * 
 */
package ceka.simulation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.Misc;

/**
 * Labeling Strategy using a single quality parameter p.
 * p is the labeling quality of a worker.
 *
 */
public class SingleQualLabelingStrategy extends LabelingStrategy {

	/**
	 * create SingleQualityParameterLabelingStrategy object.
	 * @param p the labeling quality of a worker
	 */
	public SingleQualLabelingStrategy(double p) {
		prob = p;
	}

	/* (non-Javadoc)
	 * @see ceka.simulation.LabelingStrategy#assignWorkerQuality(ceka.simulation.MockWorker[])
	 */
	@Override
	public void assignWorkerQuality(MockWorker[] workers) {
		for (int i = 0; i < workers.length; i++) {
			log.info("Worker (" + i + ") label quality = " + prob);
			workers[i].setSingleQuality(prob);
		}
	}
	
	/**
	 * @see ceka.simulation.LabelingStrategy#labelingDataset(ceka.core.Dataset)
	 */
	@Override
	public void labelDataset(Dataset dataset, MockWorker mockWorker) {
	
		ArrayList<ArrayList<Example>> exampleLists = new ArrayList<ArrayList<Example>>();
		int numCategory = dataset.getCategorySize();
		log.info ("category number: " + numCategory);
		for (int k = 0; k < numCategory; k++) {
			exampleLists.add(new ArrayList<Example>());
		}
		log.info("statistic examples belonging to each category");
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example example = dataset.getExampleByIndex(i);
			int trueCate = example.getTrueLabel().getValue();
			exampleLists.get(trueCate).add(example);
		}
		
		// calculate the correct number of labeling on each category and store the values in a list
		ArrayList<Integer> correctNumbers = new ArrayList<Integer> ();
		ArrayList<Integer> misLabelingNumbers = new ArrayList<Integer>();
		for (int k = 0; k < numCategory; k++) {
			int correct = (int)(mockWorker.getSingleQuality() * exampleLists.get(k).size());
			int incorrect = exampleLists.get(k).size() - correct;
			correctNumbers.add(new Integer(correct));
			misLabelingNumbers.add(new Integer (incorrect));
			log.info("Category " + k + " contains " + exampleLists.get(k).size() 
					+ " examples, correct labeled " + correct + " mislabeled " + incorrect);
		}
		
		Worker worker = dataset.getWorkerById(mockWorker.getId());
		if (worker == null)
			dataset.addWorker(worker = new Worker(mockWorker.getId()));
		
		// generate correct and incorrect examples
		for (int k = 0; k < numCategory; k++) {
			ArrayList<List <Example>> currExampleLists = Misc.splitRandom(exampleLists.get(k), correctNumbers.get(k).intValue());
			List<Example> correctList = currExampleLists.get(0);
			List<Example> errorList = currExampleLists.get(1);
			// process correctList
			for (Example example: correctList) {
				Label noisyLabel = new Label(null, new Integer(k).toString(), example.getId(), worker.getId());
				worker.addNoisyLabel(noisyLabel);
				example.addNoisyLabel(noisyLabel);
			}
			log.info("Category " + k + ", " + correctList.size() + " correct examples labeled");
			// process errorList
			// statistic remaining number of examples except category k
			int remainSize = 0;
			// the number of category assigning to errorList except category k
			int [] remainNumbers = new int[numCategory]; 
			for (int kk = 0; kk < numCategory; kk++) {
				if (kk != k) {
					remainSize += exampleLists.get(kk).size();
				}
			}
			// the k-2 Sum is used to prevent the sum of k-1 values not equal to errorList.size();
			int sumKMinus2 = 0;
			if (k == numCategory -1) {
				for (int k1 = 0; k1 < numCategory - 1; k1++) {
					if (k1 < numCategory - 2) {
						remainNumbers[k1] =(int) (errorList.size() * ((double) exampleLists.get(k1).size() / (double)(remainSize)));
						sumKMinus2 += remainNumbers[k1];
					} else {
						remainNumbers[k1] = errorList.size() - sumKMinus2;
					}
				}
			} else {
				for (int k1 = 0; k1 < numCategory; k1++) {
					if (k1 < numCategory - 1) {
						if (k1 != k) {
							remainNumbers[k1] =(int) (errorList.size() * ((double) exampleLists.get(k1).size() / (double)(remainSize)));
							sumKMinus2 += remainNumbers[k1];
						}
					} else {
						remainNumbers[k1] = errorList.size() - sumKMinus2;
					}
				}
			}
			// generate an error list with  number of errorList.size() elements
			ArrayList<Integer> errorCategoryList = new ArrayList<Integer>();
			for (int k2 = 0; k2 < numCategory; k2++) {
				int counter = 0;
				while (counter++ < remainNumbers[k2])
					errorCategoryList.add(new Integer(k2));
				log.info("For correct category " + k + ", Error category " + k2 + " contains " + remainNumbers[k2] + " examples");
			}
			
			// shuffle 
			assert errorCategoryList.size() == errorList.size();
			Collections.shuffle(errorCategoryList);
			int errorExampleIndex = 0;
			for (Integer errorCate : errorCategoryList) {
				Example example = errorList.get(errorExampleIndex++);
				Label noisyLabel = new Label(null, errorCate.toString(), example.getId(), worker.getId());
				worker.addNoisyLabel(noisyLabel);
				example.addNoisyLabel(noisyLabel);
			}
			log.info("For correct category " + k + ", " + errorCategoryList.size() + " error labels assigned");
		}
	}
	
	private double prob;
	private static Logger log = LogManager.getLogger(SingleQualLabelingStrategy.class);
	
}
