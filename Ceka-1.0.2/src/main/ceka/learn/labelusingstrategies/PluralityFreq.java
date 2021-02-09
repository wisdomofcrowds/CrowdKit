package ceka.learn.labelusingstrategies;

import ceka.consensus.mv.MajorityVote;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.MultiNoisyLabelSet;

public class PluralityFreq {
	
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
			Example oE = data.getExampleByIndex(i);
			Example e = (Example)oE.deepCopy();
			retData.addExample(e);
		}
		
		MajorityVote MV = new MajorityVote();
		MV.doInference(retData);
		
		for (int i = 0; i < numExample; i++) {
			double [] cateWeights = new double[retData.getCategorySize()];
			Example e = retData.getExampleByIndex(i);
			MultiNoisyLabelSet mnls = e.getMultipleNoisyLabelSet(0);
			for (int j = 0; j < mnls.getLabelSetSize(); j++)
				cateWeights [mnls.getLabel(j).getValue()]+= 1;
			// laplace correction
			for (int k = 0; k < cateWeights.length; k++)
				cateWeights[k] +=1 ;
			for (int k = 0; k < cateWeights.length; k++)
				cateWeights[k] /= (double)(mnls.getLabelSetSize()+cateWeights.length);
			e.setWeight(cateWeights[e.getIntegratedLabel().getValue()]);
		}
		
		return retData;
	}
}
