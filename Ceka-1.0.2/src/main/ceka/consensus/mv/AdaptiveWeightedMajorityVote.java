/**
 * 
 */
package ceka.consensus.mv;

import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import ceka.consensus.plat.FreqPosTable;
import ceka.consensus.plat.FreqPosTable.FreqPos;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.DescendingElement;


public class AdaptiveWeightedMajorityVote {

	private static Logger log = LogManager.getLogger(AdaptiveWeightedMajorityVote.class);
	
	public static final String NAME = "WeightedVote";
	
	public AdaptiveWeightedMajorityVote() {
		freqTable = new FreqPosTable();
	}
	
	public void doInference (Dataset dataset) {
		buildFreqTable(dataset);
		
		double thresholdFreq = estimateThresholdFreq();
		log.info("Threshold Frequency = " + thresholdFreq);
		//double biasRate = (0.5 - thresholdFreq) / 0.5;
		//double biasRate = Math.pow(1 - Math.pow(thresholdFreq / 0.5, 2), 0.5);
		double biasRate = 8 * Math.pow(thresholdFreq, 2) - 4 * thresholdFreq;
		double  c = 0.5;
		if (thresholdFreq > 0.25)
			biasRate =  8 * c * thresholdFreq - 16 * (1 - c) * Math.pow(thresholdFreq, 2);
		else
			biasRate = 8 * (c - 1) * thresholdFreq  - 16 * (c-1) * Math.pow(thresholdFreq, 2) + 1;
		log.info("Bias Rate = " + biasRate);
		double [] weights = new double[2];
		weights[0] = 0.5 * (1 - biasRate);
		weights[1] = 1 - weights[0];
		
		int numCategory = dataset.getCategorySize();
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			
			Example example = dataset.getExampleByIndex(i);
			
			ArrayList<DescendingElement <Integer>> classCountList = new ArrayList<DescendingElement<Integer>>();	
			MultiNoisyLabelSet multipleNoisyLabelSet = example.getMultipleNoisyLabelSet(0);
		    ArrayList<ArrayList<Label>> labelLists = generateLabelListByCategory(multipleNoisyLabelSet, numCategory);
		    
			for (int k = 0; k < labelLists.size(); k++) {
				
				DescendingElement <Integer> elem = new DescendingElement<Integer>();
				elem.setData(new Integer(k));
				elem.setKey(labelLists.get(k).size() * weights[k]);
				classCountList.add(elem);
			}
			
			Collections.sort(classCountList);
			
			ArrayList<DescendingElement <Integer>> maxCountList = new ArrayList<DescendingElement <Integer>>();
			maxCountList.add(classCountList.get(0));
			for (int j = 1; j < classCountList.size(); j++){
				if ((int) classCountList.get(j).getKey() == (int)classCountList.get(0).getKey())
					maxCountList.add(classCountList.get(j));
				else
					break;
			}
			Label integratedL = new Label(null, maxCountList.get(0).getData().toString(), example.getId(), NAME);
			example.setIntegratedLabel(integratedL);
		}
		// this is important
		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	private void buildFreqTable(Dataset data) {
		freqTable.buildTable(data);
	}
	
	private int lastMinimaItemsSize(ArrayList<Integer> list) {
		if (list.isEmpty())
			return 0;
		return freqTable.getFreqPos(list.get(list.size()-1).intValue()).items.size();
	}
	
	private int lastMaximaItemsSize(ArrayList<Integer> list) {
		return freqTable.getFreqPos(list.get(list.size()-1).intValue()).items.size();
	}
	
	private boolean diff(int l, int s, int N, double coefficient) {
		int e = (int) (coefficient * N / freqTable.entriesSize());
		log.debug("e = " + e);
		return ((l - s) < e) ? false : true;
	}
	
	private double estimateThresholdFreq() {
		freqTable.sort();
		
		int N = freqTable.totalItemCount();
		peakInfo = new PeakInfo();
		ArrayList<Integer> maximaSet = new ArrayList<Integer>();
		ArrayList<Integer> minimaSet = new ArrayList<Integer>();
		maximaSet.add(0);
		
		for (int i = 1; i < freqTable.entriesSize() - 1; i++) {
			FreqPos fs =  freqTable.getFreqPos(i);
			FreqPos pre = freqTable.getFreqPos(i-1);
			FreqPos pos = freqTable.getFreqPos(i+1);
			int a = fs.items.size() - pre.items.size();
			int b = pos.items.size() - fs.items.size();
			if ((a <= 0) && (b >= 0) && diff(lastMaximaItemsSize(maximaSet), fs.items.size(), N, diffCoefficient))
				minimaSet.add(i);
			if ((a >= 0) && (b <= 0) && diff(fs.items.size(), lastMinimaItemsSize(minimaSet), N, diffCoefficient))
				maximaSet.add(i);
		}
		
		// find first peak
		peakInfo.peak1 = maximaSet.get(0).intValue();
		int maxPeak = freqTable.getFreqPos(maximaSet.get(0).intValue()).items.size();
		int maxPeakIndex = 0;
		for (int i = 0; i < maximaSet.size(); i++) {
			if ((freqTable.getFreqPos(maximaSet.get(i).intValue()).items.size() >= maxPeak)
					&& (freqTable.getFreqPos(maximaSet.get(i).intValue()).freq < 0.5)) {
				peakInfo.peak1 = maximaSet.get(i).intValue();
				maxPeak = freqTable.getFreqPos(peakInfo.peak1).items.size();
				maxPeakIndex = i;
			}
		}
		log.info("First Peak Position: " + maximaSet.get(maxPeakIndex).intValue());
		
		// find second peak
		int secondPeakIndex = -1;
		int secondMax = 0;
		for (int i = maxPeakIndex + 1; i < maximaSet.size(); i++) {
			if (freqTable.getFreqPos(maximaSet.get(i).intValue()).items.size() > secondMax) {
				peakInfo.peak2 = maximaSet.get(i).intValue();
				secondMax = freqTable.getFreqPos(peakInfo.peak2).items.size();
				secondPeakIndex = i;
			}
		}
		
		// if we find peak 2
		if (secondPeakIndex != -1) {
			
			log.info("Second Peak Position:" + maximaSet.get(secondPeakIndex).intValue());
			PolynomialCurveFitter PCF = PolynomialCurveFitter.create(2);
			ArrayList<WeightedObservedPoint> points = new ArrayList<WeightedObservedPoint>();
			double grain =  1.0 / (double)(freqTable.getFreqPos(maximaSet.get(maxPeakIndex).intValue()).itemSize());
			for (int i =  maximaSet.get(maxPeakIndex).intValue(); i <= maximaSet.get(secondPeakIndex).intValue(); i++) {
				double x =  freqTable.getFreqPos(i).freq;
				double y = freqTable.getFreqPos(i).itemSize() * grain;
				points.add(new WeightedObservedPoint(1.0, x, y));
			}
			double[] coeff = PCF.fit(points);
			double vertex = - coeff[1] * 0.5 / coeff[2];
			
			if ((vertex >= 0.5) && (vertex < freqTable.getFreqPos(maximaSet.get(secondPeakIndex).intValue()).freq)) {
				log.info("find a minimum vertex > 0.5, return 0.5 as a threshold");
				return 0.5;
			}
			
			int closeToVertex = -1;
			for (int i =  maximaSet.get(maxPeakIndex).intValue(); i <= maximaSet.get(secondPeakIndex).intValue(); i++) {
				if ( freqTable.getFreqPos(i).freq >= vertex) {
					closeToVertex = i -1;
					break;
				}
			}
			if (closeToVertex == -1) {
				log.info("Cannot find a point close to vertex between the first and the second peaks. Find a minimum point left next to the second peak.");
				int nextToSecondPeak = findLessThan(minimaSet, maximaSet.get(secondPeakIndex));
				while ((nextToSecondPeak != -1) && (freqTable.getFreqPos(nextToSecondPeak).freq > 0.5))
					nextToSecondPeak = findLessThan(minimaSet, maximaSet.get(nextToSecondPeak));
				peakInfo.valley = nextToSecondPeak;
			} else {
				peakInfo.valley = closeToVertex;
			}
			
			log.info("QuadraticFitting: (a, b, c)=(" + coeff[2] + ", " + coeff[1] + ", " + coeff[0] + ") vetex=" + vertex + ", valley=" + peakInfo.valley);
		} 
		
		// determine threshold position
		int tPosition = 0;
		if (peakInfo.valley != -1)
			tPosition = peakInfo.valley;
		else
			tPosition = peakInfo.peak1;
		// fix
		int k = 0;
		for (; k <= tPosition; k++)
			peakInfo.N_L += freqTable.getFreqPos(k).items.size();
		k = tPosition;
		while ((freqTable.entriesSize() > 2) && (peakInfo.N_L < N/2) && (freqTable.getFreqPos(k + 1).freq <= 0.5)) {
			k++;
			peakInfo.N_L += freqTable.getFreqPos(k).items.size();
		}
		tPosition = k;
		double freqT =  freqTable.getFreqPos(tPosition).freq;
		
		return freqT;
	}
	
	private ArrayList<ArrayList<Label>> generateLabelListByCategory(MultiNoisyLabelSet multipleNoisyLabelSet, int numCategory) {
		ArrayList<ArrayList<Label>> labelLists= new ArrayList<ArrayList<Label>>();
		for (int k = 0; k < numCategory; k++)
			labelLists.add(new ArrayList<Label>());
		for (int i = 0; i < multipleNoisyLabelSet.getLabelSetSize(); i++) {
			Label nL =  multipleNoisyLabelSet.getLabel(i);
			labelLists.get(nL.getValue()).add(nL);
		}
		return labelLists;
	}
	
	private int findLessThan (ArrayList<Integer> list, Integer e) {
		int size = list.size();
		for (int i = size-1; i >= 0; i--) {
			if (list.get(i) < e)
				return list.get(i).intValue();
		}
		return -1;
	}
	
	private class PeakInfo {
		int peak1 = -1;
		int peak2 = -1;
		int valley = -1;
		int N_L = 0;
		@SuppressWarnings("unused")
		int N_R = 0;
	}
	
	private FreqPosTable       freqTable;
	private PeakInfo peakInfo;
	
	private static final double diffCoefficient = 0.03;
	
}
