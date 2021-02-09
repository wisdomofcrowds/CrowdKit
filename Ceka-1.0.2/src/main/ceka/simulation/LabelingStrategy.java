/**
 * 
 */
package ceka.simulation;

import ceka.core.Dataset;

/**
 * The abstract class of Labeling Strategy
 *
 */
public abstract class LabelingStrategy {
	
	/**
	 * enumerated code of labeling strategy
	 */
	public static enum Code {
		/**
		 * single quality parameter labeling strategy
		 */
		LS_SINGLE_QUALITY,
		/**
		 * Gaussian quality parameter labeling strategy
		 */
		LS_GAUSSIAN_QUALITY
	}
	
	/**
	 * assign qualities to workers
	 * @param workers
	 */
	public abstract void assignWorkerQuality(MockWorker [] workers);
	
	/**
	 * label a dataset
	 * @param dataset the dataset to be labeled
	 * @param worker  the worker labeling the dataset
	 */
	public abstract void labelDataset(Dataset dataset, MockWorker worker);
	
}
