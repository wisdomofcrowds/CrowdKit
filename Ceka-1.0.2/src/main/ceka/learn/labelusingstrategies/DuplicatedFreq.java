package ceka.learn.labelusingstrategies;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.MultiNoisyLabelSet;

public class DuplicatedFreq {
	
	public static Dataset makeDataset(Dataset data) {
		Dataset retData = data.generateEmpty();
		
		// see how many classes
		int numCategory = data.getCategorySize();
		// copy the categories to two newly created data sets
		for (int i = 0; i < numCategory; i++) {
			Category cate = data.getCategory(i);
			retData.addCategory(cate.copy());
		}
		
		int numExample = data.getExampleSize();
		
		for (int i = 0; i < numExample; i++) {
			int [] cateCount = new int[data.getCategorySize()];
			Example oE = data.getExampleByIndex(i);
			MultiNoisyLabelSet mnls = oE.getMultipleNoisyLabelSet(0);
			for (int j = 0; j < mnls.getLabelSetSize(); j++)
				cateCount [mnls.getLabel(j).getValue()]++;
			// laplace correction
			for (int k = 0; k < cateCount.length; k++)
				cateCount[k]++;
			// make duplicated instances
			for (int k = 0; k < cateCount.length; k++) {
				Example e = (Example)oE.deepCopy();
				e.setWeight((double)cateCount[k] / (double)(mnls.getLabelSetSize()+cateCount.length));
				e.setClassValue(k);
				retData.addExample(e);
			}
		}
		
		return retData;
	}

}
