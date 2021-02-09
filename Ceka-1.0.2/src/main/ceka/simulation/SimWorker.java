/**
 * 
 */
package ceka.simulation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import ceka.core.Category;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.Misc;

/**
 * @author Jing
 * A super class of simulated workers
 *
 */
public class SimWorker  extends Worker{

	/**
	 * @param id worker identifier
	 */
	public SimWorker(String id) {
		super(id);
	}

	/**
	 *  the overall quality of the worker
	 * @return
	 */
	public double getQuality() {
		return q;
	}
	
	public void setQuality(double v) {
		q = v;
	}
	
	/**
	 * On each category, the correct rate is p. The category distribution of error labels on each category is
	 * calculated by the proportion of the error category across all error labels.
	 * @param exampleList
	 * @param cateList
	 */
	public void uniformLabeling(List<Example> exampleList, List<Category> cateList) {
		ArrayList<ArrayList<Example>> exampleLists = new ArrayList<ArrayList<Example>>();
		int numCate = cateList.size();
		for (int k = 0; k < numCate; k++) {
			exampleLists.add(new ArrayList<Example>());
		}
		for (Example e : exampleList) {
			int trueClass = e.getTrueLabel().getValue();
			exampleLists.get(trueClass).add(e);
		}
		
		// calculate the correct number of labeling on each category and store the values in a list
		ArrayList<Integer> correctNumbers = new ArrayList<Integer> ();
		ArrayList<Integer> wrongNumbers = new ArrayList<Integer>();
		for (int k = 0; k < numCate; k++) {
			int correct = (int)(q * exampleLists.get(k).size());
			int incorrect = exampleLists.get(k).size() - correct;
			correctNumbers.add(new Integer(correct));
			wrongNumbers.add(new Integer (incorrect));
			log.info("Category (" + k + ") contains " + exampleLists.get(k).size() 
					+ " examples (" + correct + " correct and " + incorrect +" incorrect).");
		}
		
		// generate correct and incorrect examples
		for (int k = 0; k < numCate; k++) {
			ArrayList<List <Example>> currExampleLists = Misc.splitRandom(exampleLists.get(k), correctNumbers.get(k).intValue());
			List<Example> correctList = currExampleLists.get(0);
			List<Example> wrongList = currExampleLists.get(1);
			// process correctList
			for (Example example: correctList) {
				Label noisyLabel = new Label(null, new Integer(k).toString(), example.getId(), getId());
				addNoisyLabel(noisyLabel);
				example.addNoisyLabel(noisyLabel);
			}
			log.info("Category (" + k + ") has " + correctList.size() + " correct examples labeled");
			// process errorList
			// count remaining number of examples except category k
			int remainSize = 0;
			// the number of category assigning to errorList except category k
			int [] remainNumbers = new int[numCate]; 
			for (int r = 0; r < numCate; r++) {
				if (r != k) {
					remainSize += exampleLists.get(r).size();
				}
			}
			// the k-2 Sum is used to prevent the sum of k-1 values not equal to errorList.size();
			int sumKMinus2 = 0;
			if (k == numCate - 1) {
				for (int k1 = 0; k1 < numCate - 1; k1++) {
					if (k1 < numCate - 2) {
						remainNumbers[k1] =(int) (wrongList.size() * ((double) exampleLists.get(k1).size() / (double)(remainSize)));
						sumKMinus2 += remainNumbers[k1];
					} else {
						remainNumbers[k1] = wrongList.size() - sumKMinus2;
					}
				}
			} else {
				for (int k1 = 0; k1 < numCate; k1++) {
					if (k1 < numCate - 1) {
						if (k1 != k) {
							remainNumbers[k1] =(int) (wrongList.size() * ((double) exampleLists.get(k1).size() / (double)(remainSize)));
							sumKMinus2 += remainNumbers[k1];
						}
					} else {
						remainNumbers[k1] = wrongList.size() - sumKMinus2;
					}
				}
			}
			// generate an error list with  number of errorList.size() elements
			ArrayList<Integer> errorCategoryList = new ArrayList<Integer>();
			for (int k2 = 0; k2 < numCate; k2++) {
				int counter = 0;
				while (counter++ < remainNumbers[k2])
					errorCategoryList.add(new Integer(k2));
				log.info("For correct category (" + k + "), error category (" + k2 + ") contains " + remainNumbers[k2] + " examples");
			}
			
			// shuffle 
			assert errorCategoryList.size() == wrongList.size();
			Collections.shuffle(errorCategoryList);
			int errorExampleIndex = 0;
			for (Integer errorCate : errorCategoryList) {
				Example example = wrongList.get(errorExampleIndex++);
				Label noisyLabel = new Label(null, errorCate.toString(), example.getId(), getId());
				addNoisyLabel(noisyLabel);
				example.addNoisyLabel(noisyLabel);
			}
			log.info("For correct category (" + k + "), " + errorCategoryList.size() + " error labels assigned");
		}
	}
	
	protected double q = 0.5; /* 0.5 means random guess */
	
	protected static Logger log = LogManager.getLogger(SimWorker.class);
}
