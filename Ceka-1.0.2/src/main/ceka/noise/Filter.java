/**
 * 
 */
package ceka.noise;

import java.util.ArrayList;

import weka.classifiers.Classifier;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.utils.Misc;

/**
 * abstract class for all filter
 *
 */
public abstract class Filter {

	public Dataset getCleansedDataset() {
		return cleanedDataset;
	}
	
	public Dataset getNoiseDataset() {
		return noiseDataset;
	}
	
	/**
	 * filtering dataset to form two sub dataset cleaned dataset and noise dataset
	 * @param dataset
	 * @param classifer
	 * @throws Exception 
	 */
	public abstract void filterNoise(Dataset dataset, Classifier[] classifer) throws Exception;
	
	/**
	 * get the number of true noise if the ground truth (true label) is known
	 * this method must be called after calling fiterNoise, otherwise -1 will return
	 * @return the number of true noise
	 */
	public int getNumberTrueNoise() {
		statistic();
		return nTrueNoise;
	}
	
	/**
	 * get the recall value of the predicted noises
	 * this method must be called after calling fiterNoise, otherwise -1 will return
	 * @return the recall value
	 */
	public double getRecall() {
		statistic();
		return recall;
	}
	
	/**
	 * get the precise value of the predicted noises
	 * this method must be called after calling fiterNoise, otherwise -1 will return
	 * @return the precise value
	 */
	public double getPrecise() {
		statistic();
		return precise;
	}
	
	/**
	 * get the f1-score value of the predicted noises
	 * this method must be called after calling fiterNoise, otherwise -1 will return
	 * @return the f1-score value
	 */
	public double getF1Score() {
		statistic();
		return f1Score;
	}
	
	/**
	 * get filter statistic information
	 * @return
	 */
	public String getStatisticInfo() {
		statistic();
		double totalNum = (double) originalDataset.getExampleSize();
		double predictFraction = (double) noiseDataset.getExampleSize() / totalNum;
		double trueFraction = (double) nTrueNoise / totalNum;
		String str = "Predict " + noiseDataset.getExampleSize() + " noise (" + predictFraction + ") | Recall: " + recall 
				+ " | Precise: " + precise + " | F1: "+ f1Score +" | true noise " + nTrueNoise + " (" + trueFraction+ ")";
		return str;
	}
	
	protected void createInternal(Dataset dataset) {
		cleanedDataset = dataset.generateEmpty();
		noiseDataset = dataset.generateEmpty();
		int numCateSize = dataset.getCategorySize();
		for (int i = 0; i < numCateSize; i++) {
			Category cate = dataset.getCategory(i);
			cleanedDataset.addCategory(cate.copy());
			noiseDataset.addCategory(cate.copy());
		}
		originalDataset = dataset;
		statFlag = false;
	}
	
	protected void statistic() {
		if (!statFlag) {
			ArrayList<Example> trueNoiseList = new ArrayList<Example>();
			for (int i = 0; i < originalDataset.getExampleSize(); i++) {
				Example example = originalDataset.getExampleByIndex(i);
				if (((int)example.value(example.numAttributes() - 1)) != example.getTrueLabel().getValue()) {
					trueNoiseList.add(example);
				}
			}
			nTrueNoise = trueNoiseList.size();
			int correctNum = 0;
			for (int i= 0; i < noiseDataset.getExampleSize(); i++) {
				Example example = noiseDataset.getExampleByIndex(i);
				if (Misc.getElementById(trueNoiseList, example.getId()) != null)
					correctNum += 1;
			}
			recall = (double)correctNum / (double) trueNoiseList.size();
			precise = (double)correctNum / (double) noiseDataset.getExampleSize();
			f1Score = 2* recall * precise / (recall + precise);
		}
		statFlag = true;
	}
	
	protected Dataset cleanedDataset = null;
	protected Dataset noiseDataset = null;
	
	private int nTrueNoise = -1;
	private double recall = -1;
	private double precise = -1;
	private double f1Score = -1;
	protected Dataset originalDataset = null;
	protected boolean statFlag = false;
}
