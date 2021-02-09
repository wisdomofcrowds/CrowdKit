/**
 * 
 */
package ceka.simulation;

import ceka.core.Dataset;
import ceka.core.Worker;

/**
 * A simulant worker.
 *
 */
public class MockWorker extends Worker {

	/**
	 * @param id worker identifier
	 */
	public MockWorker(String id) {
		super(id);
	}
	
	/**
	 * label a data set with specific a labeling strategy
	 * @param dataset
	 * @param strategy {@link LabelingStrategy}
	 */
	public void labeling(Dataset dataset, LabelingStrategy strategy) {
		  strategy.labelDataset(dataset, this);
	}
	
	public void setSingleQuality(double q) {
		singleQuality = q;
	}
	
	public double getSingleQuality() {
		return singleQuality;
	}
	
	// used for single quality model
	private double singleQuality = 0.0;
}
