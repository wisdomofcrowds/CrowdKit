/**
 * 
 */
package ceka.noise.avnc;

import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Classifier;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.noise.Filter;
import ceka.utils.AscendingElement;
import ceka.utils.DatasetManipulator;
import ceka.utils.IdDecorated;
import ceka.utils.Misc;
import ceka.utils.PerformanceStatistic;

/**
 * A novel classification filter for crowdsourcing learning
 * Citation: Jing Zhang, Victor S. Sheng, Tao Li, & Xindong Wu. (Online). 
 *           Improving Crowdsourced Label Quality Using Noise Correction.
 *           IEEE Transactions on Neural Networks and Learning Systems
 */
public class AdaptiveClassificationFilter extends Filter {

	/**
	 * Create an AdaptiveClassificationFilter
	 * @param nFolds the data set will be splitted into nFold.
	 * @param nClassifier number of ensemble classifier
	 */
	public AdaptiveClassificationFilter(int nFolds, int nClassifiers) {
		this.nFolds = nFolds;
		this.nModel = nClassifiers;
		this.threshold = (int) Math.ceil((double)this.nModel / 2);
		highQLDatasets = new Dataset[this.nModel];
	}
	
	@Override
	public void filterNoise(Dataset dataset, Classifier[] classifier) throws Exception {
		createInternal(dataset);
		// run classification filter nClassifiers times
		for (int i = 0; i < nModel; i++) {
			highQLDatasets[i] = filterOnce(dataset, classifier[0], i);
		}
		determineNoise();
	}
	
	/**
	 * get high quality data sets
	 * @return the high quality data sets
	 */
	public Dataset [] getHighQualityDatasets() {
		return highQLDatasets;
	}
	
	/**
	 * set the minimum proportion of noise estimated
	 * @param pro
	 */
	public void setMinEstimatedNoiseProportion(double pro) {
		minNoiseProportion = pro;
	}
	
	/**
	 * set the maximum proportion of noise estimated
	 * @param pro
	 */
	public void setMaxEstimatedNoiseProportion(double pro) {
		maxNoiseProportion = pro;
	}
	
	/**
	 * set the threshold that the number of models determining a noise
	 * @param threshold
	 */
	public void setNoiseDeterminationThreshold(int threshold) {
		this.threshold = threshold;
	}
	
	private Dataset filterOnce (Dataset dataset, Classifier classifier, int round) throws Exception{
		Dataset highQLDataset = dataset.generateEmpty();
		Dataset lowQLDataset = dataset.generateEmpty();
		int numCateSize = dataset.getCategorySize();
		for (int i = 0; i < numCateSize; i++) {
			Category cate = dataset.getCategory(i);
			highQLDataset.addCategory(cate.copy());
			lowQLDataset.addCategory(cate.copy());
		}
		Dataset [] splittedDatasets = DatasetManipulator.split(dataset, nFolds, true);
		for (int i = 0; i < nFolds; i++) {
			Dataset [] trainAndTest = DatasetManipulator.pickCombine(splittedDatasets, i);
			Dataset trainDataset = trainAndTest[0];
			Dataset testDataset = trainAndTest[1];
			// train a model
			classifier.buildClassifier(trainDataset);
			// test one by one
			for (int j = 0; j < testDataset.getExampleSize(); j++) {
				Example testExample = testDataset.getExampleByIndex(j);
				// recode info
				Ballot ballot = null;
				ballot = getBallot(testExample.getId());
				if (ballot == null){
					// new ballot
					ballot = new Ballot(testExample);
					ballotTable.add(ballot);
				}
				double predict = classifier.classifyInstance(testExample);
				double [] distrib =classifier.distributionForInstance(testExample);
				int same = -1;
				if ((int)predict != testExample.getTrainingLabel()) {
					lowQLDataset.addExample(testExample);
					same = 0;
				} else {
					highQLDataset.addExample(testExample);
					same = 1;
				}
				// and a vote info.
				PredictedInfo predictedInfo = new PredictedInfo();
				predictedInfo.classDist = distrib;
				predictedInfo.round = round;
				predictedInfo.category = (int) predict;
				predictedInfo.same = same;
				ballot.modelVotes.add(predictedInfo);
				//System.out.println("Round " + round + " Fold " + i + " Example " + testExample.getId());
			}
		}
		//PerformanceStatistic perfStat = new PerformanceStatistic();
		//perfStat.stat(highQLDataset);
		//System.out.println(" length = " + highQLDataset.getExampleSize() +" accuracy = " + perfStat.getAccuracy());
		return highQLDataset;
	}
	
	private void determineNoise () {
		ArrayList<ArrayList <AscendingElement<Ballot>>> layerBallotLists = new ArrayList<ArrayList<AscendingElement<Ballot>>>();
		for (int i = 0; i<= nModel; i++)
			layerBallotLists.add(new ArrayList<AscendingElement<Ballot>>());
		
		// information statistic
		for (Ballot ballot : ballotTable) {
			ballot.stat();
			AscendingElement<Ballot> ae = new AscendingElement<Ballot>();
			ae.setData(ballot);
			ae.setKey(ballot.entropy);
			layerBallotLists.get(ballot.numSame).add(ae);
		}
		
		// sort in Ascending order of entropy
		for (int i = 0; i <= nModel; i++){
			Collections.sort(layerBallotLists.get(i));
			//statLayeredListPerf(layerBallotLists.get(i), i);
		}
		
		int minNumberNoise = (int) (minNoiseProportion * ballotTable.size());
		int maxNumberNoise = (int) (maxNoiseProportion * ballotTable.size());
		int thresholdNum = 0;
		for (int i = 0; i <= (nModel-threshold); i++) {
			thresholdNum += layerBallotLists.get(i).size();
		}
		
		if (thresholdNum < maxNumberNoise) {
			maxNumberNoise = thresholdNum;
		}
		
		if (maxNumberNoise < minNumberNoise)
			maxNumberNoise = minNumberNoise;
		
		int layerIndex = 0;
		int layerCount = 0;
		int totalCount = 0;
		while (totalCount < ballotTable.size()) {
			if ( layerBallotLists.get(layerIndex).isEmpty()) {
				layerIndex += 1;
				layerCount = 0;
				continue;
			}
			if (layerCount == layerBallotLists.get(layerIndex).size()) {
				layerIndex += 1;
				layerCount = 0;
				continue;
			}
			AscendingElement<Ballot> ae = layerBallotLists.get(layerIndex).get(layerCount);
			if (totalCount < maxNumberNoise)
				noiseDataset.addExample(ae.getData().example);
			else
				cleanedDataset.addExample(ae.getData().example);
			layerCount++;
			totalCount++;
		}
	}
	
	private void statLayeredListPerf(ArrayList <AscendingElement<Ballot>> list, int seqList) {
		Dataset data = new Dataset(highQLDatasets[0], 0);
		int numCateSize = highQLDatasets[0].getCategorySize();
		for (int i = 0; i < numCateSize; i++) {
			Category cate = highQLDatasets[0].getCategory(i);
			data.addCategory(cate.copy());
		}
		for (AscendingElement<Ballot> ae: list) {
			data.addExample(ae.getData().example);
		}
		if (list.size() > 0) {
			PerformanceStatistic perfStat = new PerformanceStatistic();
			perfStat.stat(data);
			System.out.println("list " + seqList + " length = " + list.size() +" accuracy = " + perfStat.getAccuracy());
		}
	}
	
	//////////////////////////////////////////////////////////////////////////////
	private class PredictedInfo {
		public double [] classDist = null;
		public int category = -1; // predicted category
		public int same     = -1; // 1 means the classified equals to the original label, 0 means different.
		public int round    = -1;
	}
	
	private class Ballot implements IdDecorated {
	
		public Ballot (Example e) {
			this.example = e;
		}
		public Example example = null;
		public ArrayList<PredictedInfo> modelVotes = new ArrayList<PredictedInfo>();
		/* (non-Javadoc)
		 * @see ceka.utils.IdDecorated#getId()
		 */
		@Override
		public String getId() {
			return example.getId();
		}
		public void stat () {
			for (PredictedInfo info: modelVotes) {
				if (info.same == 1)
					numSame += 1;
				for (int i = 0; i < info.classDist.length; i++) {
					if (Misc.isDoubleSame(info.classDist[i], 0, 0.0000001))
						entropy -= 0;
					else
						entropy -=  (info.classDist[i] * Math.log(info.classDist[i]));
				}
			}
		}
		public int numSame = 0; // number of votes that the same as the original label
		public double entropy = 0.0;
	}
	
	private Ballot getBallot(String id) {
		Ballot ballot = null;
		ballot = Misc.getElementById(ballotTable, id);
		return ballot;
	}
	///////////////////////////////////////////////////////////////////////////////////
	
	private int nFolds = 5;
	private int nModel = 5;
	private ArrayList<Ballot> ballotTable = new ArrayList<Ballot>(); // each example has a ballot
	private Dataset[] highQLDatasets = null;	// High Quality Datasets
	private double minNoiseProportion = 0.0;
	private double maxNoiseProportion = 0.0;
	private int    threshold = nModel;
}
