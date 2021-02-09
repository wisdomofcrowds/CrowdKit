package ceka.consensus.iwmv;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;
import ceka.utils.DescendingElement;

/**
 * paper Error Rate Bounds and Iterative Weighted Majority Voting
for Crowdsourcing
 * @author Jing
 *
 */

public class IWMV {
	
	public static final String NAME = "IWMV";
	
	public IWMV(int maxIter) {
		this.maxIteration = maxIter;
	}
	
	public void doInference(Dataset dataset) {
		int numWorkers  = dataset.getWorkerSize();
		
		// initialize the variables
		for (int i = 0; i < numWorkers; i++) {
			v.put(dataset.getWorkerByIndex(i).getId(), new Double(1.0));
			w.put(dataset.getWorkerByIndex(i).getId(), new Double(0.0));
		}
		
		int iter = 0;
		while (iter++ < maxIteration) {
			 mvWithV(dataset);
			 calculateW(dataset);
			 calculateV(dataset);
		}
	}
	
	private void mvWithV(Dataset dataset) {
		
		int numCategory = dataset.getCategorySize();
		
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			ArrayList<DescendingElement <Integer>> classCountList = new ArrayList<DescendingElement<Integer>>();
			Example example = dataset.getExampleByIndex(i);
			
			MultiNoisyLabelSet multipleNoisyLabelSet = example.getMultipleNoisyLabelSet(0);
		    ArrayList<ArrayList<Label>> labelLists = generateLabelListByCategory(multipleNoisyLabelSet, numCategory);
		    
			for (int k = 0; k < labelLists.size(); k++) {
				DescendingElement <Integer> elem = new DescendingElement<Integer>();
				elem.setData(new Integer(k));
				// here we consider v_i
				int listLen = labelLists.get(k).size();
				if (listLen == 0) {
					elem.setKey(new Double(0));
				} else {
					double sum = 0;
					for (int q = 0; q < listLen; q++) {
						Label label = labelLists.get(k).get(q);
						String wId = label.getWorkerId();
						sum += v.get(wId).doubleValue();
					}
					elem.setKey(sum);
				}
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
			
			int s = 0;
			if (maxCountList.size() > 1) {
				double r = Math.random();
				double grain = 1.0 / (double)maxCountList.size();
				s =  (int)(r / grain);
			}
			
			Label integratedL = new Label(null, maxCountList.get(s).getData().toString(), example.getId(), NAME);
			example.setIntegratedLabel(integratedL);
		}
		// this is important
		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	private void calculateW(Dataset dataset) {
		int numWorkers  = dataset.getWorkerSize();
		for (int i = 0; i < numWorkers; i++) {
			Worker worker = dataset.getWorkerByIndex(i);
			MultiNoisyLabelSet mnls = worker.getMultipleNoisyLabelSet(0);
			double correct  = 0;
			int totalLables = mnls.getLabelSetSize();
			for (int q = 0; q < totalLables; q++) {
				Label label = mnls.getLabel(q);
				String instId = label.getExampleId();
				Example e = dataset.getExampleById(instId);
				if (e.getIntegratedLabel().getValue() == label.getValue()) {
					correct += 1;
				}
			}
			w.put(worker.getId(), new Double(correct/totalLables));
		}
	}
	
	private void calculateV(Dataset dataset) {
		int numWorkers  = dataset.getWorkerSize();
		int numCate = dataset.getCategorySize();
		for (int i = 0; i < numWorkers; i++) {
			Worker worker = dataset.getWorkerByIndex(i);
			Double wi = w.get(worker.getId());
			double sum = 0;
			for (int k = 0; k < numCate; k++) {
				int cate = dataset.getCategorySet(0).get(k).getValue();
				sum += wi*(cate+1);
			}
			sum -= 1;
			v.put(worker.getId(), new Double(sum));
		}
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
	
	private HashMap<String, Double> v = new HashMap<String, Double>();
	private HashMap<String, Double> w = new HashMap<String, Double>();
	private int maxIteration = 50;
}
