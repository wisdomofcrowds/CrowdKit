package ceka.consensus.mv;

import java.util.ArrayList;
import java.util.Collections;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.DescendingElement;

/**
 * The simplest Majority Voting algorithm.
 * @author Zhang
 *
 */
public class MajorityVote {
	
	public static final String NAME = "MV";
	
	/**
	 * inference the integrated label of each example in dataset
	 * @param dataset
	 */
	public void doInference(Dataset dataset) {
		
		int numCategory = dataset.getCategorySize();
		
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			ArrayList<DescendingElement <Integer>> classCountList = new ArrayList<DescendingElement<Integer>>();
			Example example = dataset.getExampleByIndex(i);
			
			MultiNoisyLabelSet multipleNoisyLabelSet = example.getMultipleNoisyLabelSet(0);
		    ArrayList<ArrayList<Label>> labelLists = generateLabelListByCategory(multipleNoisyLabelSet, numCategory);
		    
			for (int k = 0; k < labelLists.size(); k++) {
				DescendingElement <Integer> elem = new DescendingElement<Integer>();
				elem.setData(new Integer(k));
				elem.setKey(labelLists.get(k).size());
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
}
