/**
 * 
 */
package ceka.noise.avnc;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import weka.classifiers.Classifier;
import ceka.core.Dataset;
import ceka.core.Example;

/**
 * @author Zhang
 *
 */
public class VoteCorrection {
	
	public Dataset correct(Dataset noiseData, Dataset [] modelData, Classifier[] classifiers, int threshold) throws Exception {
		if (modelData.length != classifiers.length)
			return null;
		
		Classifier [] correctClassifiers = new Classifier[classifiers.length];
		for (int i = 0; i < classifiers.length; i++) {
			classifiers[i].buildClassifier(modelData[i]);
			correctClassifiers[i] = classifiers[i];
		}
		
		for (int i = 0; i < noiseData.numInstances(); i++) {
			
			Example noise = noiseData.getExampleByIndex(i);
			
			for (int k = 0; k < classifiers.length; k++) {
				if (modelData[k].getExampleById(noise.getId()) != null) {
					// remove this data
					log.info("find noise <" + noise.getId() + "> in model data " + k);
					Dataset newData = modelData[k].generateEmpty();
					for (int j = 0; j < modelData[k].getExampleSize(); j++) {
						if (!modelData[k].getExampleByIndex(j).getId().equals(noise.getId()))
							newData.add(modelData[k].getExampleByIndex(j));
					}
					correctClassifiers[k].buildClassifier(newData); 
				} else {
					correctClassifiers[k] = classifiers[k];
				}
			}
			
			int [] dist = new int[noiseData.getCategorySize()]; 
			for (int j = 0; j < correctClassifiers.length; j++) {
				int predict = (int)correctClassifiers[j].classifyInstance(noise);
					dist[predict] += 1;
			}
			int max = 0;
		    int maxIndex = 0;  
		    for (int k = 0; k < dist.length; k++) {
		    	if (dist[k] > max) {
		    		maxIndex = k;
		    		max = dist[k];
		    	}
		      }
		    // flip
		    if (dist[maxIndex] >= threshold) {
		    	noise.getIntegratedLabel().setValue(maxIndex);
		    	noise.setTrainingLabel(maxIndex);
		    }
		}
		return noiseData;
	}
	
	public Dataset correctFast(Dataset noiseData, Dataset [] modelData, Classifier[] classifiers, int threshold) throws Exception {
		if (modelData.length != classifiers.length)
			return null;
		
		Dataset[] newModelDatasets = new Dataset[classifiers.length];
		
		for (int i = 0; i < classifiers.length; i++) {
			newModelDatasets[i] =  modelData[i].generateEmpty();
			for (int j = 0; j < modelData[i].getExampleSize(); j++) {
				if (noiseData.getExampleById(modelData[i].getExampleByIndex(j).getId()) == null)
					newModelDatasets[i].addExample(modelData[i].getExampleByIndex(j));
				else
					log.info("find noise <" + modelData[i].getExampleByIndex(j).getId() + "> in model data " + i);
			}
			classifiers[i].buildClassifier(newModelDatasets[i]);
		}
		
		for (int i = 0; i < noiseData.numInstances(); i++) {
			Example noise = noiseData.getExampleByIndex(i);			
			int [] dist = new int[noiseData.getCategorySize()]; 
			for (int j = 0; j < classifiers.length; j++) {
				int predict = (int)classifiers[j].classifyInstance(noise);
					dist[predict] += 1;
			}
			int max = 0;
		    int maxIndex = 0;  
		    for (int k = 0; k < dist.length; k++) {
		    	if (dist[k] > max) {
		    		maxIndex = k;
		    		max = dist[k];
		    	}
		      }
		    // flip
		    if (dist[maxIndex] >= threshold) {
		    	noise.getIntegratedLabel().setValue(maxIndex);
		    	noise.setTrainingLabel(maxIndex);
		    }
		}
		
		return noiseData;
	}
	
	private static Logger log = LogManager.getLogger(VoteCorrection.class);
}
