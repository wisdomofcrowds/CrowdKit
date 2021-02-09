package ceka.consensus.plat;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.ListIterator;

import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import ceka.consensus.plat.FreqPosTable.FreqPos;
import ceka.core.Dataset;

public class PLATCore {
	
	private static Logger log = LogManager.getLogger(PLATCore.class);
	
	public FreqPosTable       freqTable;
	public ArrayList<Integer> posLabelExamples;
	public ArrayList<Integer> negLabelExamples;
	public int 				  thresholdPosition;
	
	public PLATCore() {
		freqTable = new FreqPosTable();
		posLabelExamples = new ArrayList<Integer>();
		negLabelExamples = new ArrayList<Integer>();
	}
	
	public void buildFreqTable(Dataset data) {
		freqTable.buildTable(data);
	}
	
	public void thresholdMethod() {
		categorying(estimateThresholdPosition());
	}
	
	public void setUseQuadraticFitting(boolean flag) {
		useQuadraticFitting = flag;
	}
	
	public double getThresholdFreq () {
		return freqTable.getFreqPos(thresholdPosition).freq;
	}
	
	public void printInfo(FileWriter file) {
		try {
			freqTable.printTable(file);
			
			String pstr = null;
			ListIterator<Integer> lbiter = posLabelExamples.listIterator();
			file.write("POS LABEL:  ");
			for (int i = 0; i < posLabelExamples.size(); i++) {
				pstr = String.format("%d ", lbiter.next().intValue());
				file.write(pstr);
			}
			file.write("\n");
			lbiter = negLabelExamples.listIterator();
			file.write("NEG LABEL:  ");
			for (int i = 0; i < negLabelExamples.size(); i++) {
				pstr = String.format("%d ", lbiter.next().intValue());
				file.write(pstr);
			}
			file.write("\n");	
		} catch (IOException e) {
			e.printStackTrace();
		}
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
	
	private int estimateThresholdPosition() {
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
			if (!useQuadraticFitting) {
				int minvalley = maxPeak;
				for (int j = 0; j < minimaSet.size(); j++) {
					if ((freqTable.getFreqPos(minimaSet.get(j).intValue()).items.size() < minvalley) 
							&& (minimaSet.get(j).intValue() > peakInfo.peak1)
							&& (minimaSet.get(j).intValue() < peakInfo.peak2)
							&& (freqTable.getFreqPos(minimaSet.get(j).intValue()).freq < 0.5)) {
						peakInfo.valley = minimaSet.get(j).intValue();
						minvalley = freqTable.getFreqPos(peakInfo.valley).items.size();
					}
				}
				// if we cannot find a point < 0.5 between to peaks, then we find a point less than but the most close to 0.5
				if (peakInfo.valley == -1) {
					for (int i = 0; i < freqTable.entriesSize(); i++) {
						if (freqTable.getFreqPos(i).freq > 0.5) {
							peakInfo.valley = i - 1;
							break;
						}
					}
				}
			} else { // second method fitting quadratic curve
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
		}
		
		// determine threshold position
		int tPosition = 0;
		if (peakInfo.valley != -1)
			tPosition = peakInfo.valley;
		else
			tPosition = peakInfo.peak1;
		int k = 0;
		for (; k <= tPosition; k++)
			peakInfo.N_L += freqTable.getFreqPos(k).items.size();
		k = tPosition;
		while ((freqTable.entriesSize() > 2) && (peakInfo.N_L < N/2)) {
			k++;
			peakInfo.N_L += freqTable.getFreqPos(k).items.size();
		}
		tPosition = k;
		peakInfo.N_R = N - peakInfo.N_L;
		
		thresholdPosition = tPosition;
		
		log.info("(peak1=" + peakInfo.peak1 + " peak2=" + peakInfo.peak2 + " valley=" + peakInfo.valley +
				" Threshold Position=" +  tPosition + " frequency=" + freqTable.getFreqPos(tPosition).freq);
		
		return thresholdPosition;
	}
	
	private void categorying(int tPosition)
	{	
		int maxPositive = ((peakInfo.N_L - peakInfo.N_R) * peakInfo.N_R / (peakInfo.N_L+ peakInfo.N_R)) + peakInfo.N_R;
		log.info("N_L=" + peakInfo.N_L + " N_R=" + peakInfo.N_R + " Max_P=" + maxPositive);
		
		int k = freqTable.entriesSize() - 1;
		int numPositive = 0;
		
		while (k > tPosition)
		{
			freqTable.getFreqPos(k).category = 1;
			numPositive += freqTable.getFreqPos(k).items.size();
			k--;
		}
		
		double theta = 0.5;
		double fmid = (freqTable.getFreqPos(0).freq +  freqTable.getFreqPos(tPosition).freq) * theta;
		
		k = tPosition;
		while ((freqTable.getFreqPos(k).freq > fmid) && ((numPositive + freqTable.getFreqPos(k).items.size()) < maxPositive)) {
			freqTable.getFreqPos(k).category = 1;
			numPositive += freqTable.getFreqPos(k).items.size();
			k--;
		}
		
		for (int i = 0; i < freqTable.entriesSize(); i++) {
			FreqPos fs = freqTable.getFreqPos(i);
			if (fs.category == 1)
				addItemToList(posLabelExamples, fs);
			else
				addItemToList(negLabelExamples, fs);
		}
	}
	
	private int addItemToList(ArrayList<Integer> list, FreqPos fs) {
		int size = fs.items.size();
		for (int i = 0; i < size; i++) {
			Integer e = fs.items.get(i);
			list.add(e);
		}
		return size;
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
		int N_R = 0;
	}
	
	private PeakInfo peakInfo;
	private boolean useQuadraticFitting = false;
	
	private static final double diffCoefficient = 0.03;
}
