/**
 * 
 */
package ceka.converters;

import java.io.File;
import java.io.FileWriter;
import java.util.HashMap;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

/**
 * Covert dataset to the file that can be processed by GLAD
 * @author Zhang
 *
 */
public class GLADConverter {

	/**
	 * convert dataset to the file can be processed by GLAD
	 * @param dataset
	 * @param gladPath
	 * @throws Exception
	 */
	public void saveDataset(Dataset dataset, String gladPath) throws Exception {
		FileWriter gladFile = new FileWriter(new File(gladPath));
		
		int numWorker = dataset.getWorkerSize();
		int totalNoisyLabels = 0;
		for (int i = 0; i < numWorker; i++) {
			Worker w = dataset.getWorkerByIndex(i);
			workerId2Index.put(w.getId(), new Integer(i).toString());
			workerIndex2Id.put(new Integer(i).toString(), w.getId());
			MultiNoisyLabelSet mnls = w.getMultipleNoisyLabelSet(0);
			totalNoisyLabels += mnls.getLabelSetSize();
		}
		
		gladFile.write(Integer.toString(totalNoisyLabels)+ " "+Integer.toString(numWorker)+" "+Integer.toString(dataset.getExampleSize())+" 0.5");
		
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example e = dataset.getExampleByIndex(i);
			exampleId2Index.put(e.getId(), new Integer(i).toString());
			exampleIndex2Id.put(new Integer(i).toString(), e.getId());
			MultiNoisyLabelSet mnls = e.getMultipleNoisyLabelSet(0);
			for (int j = 0; j < mnls.getLabelSetSize(); j++) {
				Label label = mnls.getLabel(j);
				gladFile.write("\n" + exampleId2Index.get(e.getId()) + " " + workerId2Index.get(label.getWorkerId()) + " " + label.getValue());
			}
		}
		gladFile.close();
	}
	
	public String getExampleId(String index) {
		return exampleIndex2Id.get(index);
	}
	
	private HashMap<String, String> exampleId2Index = new HashMap<String, String>();
	private HashMap<String, String> exampleIndex2Id = new HashMap<String, String>();
	private HashMap<String, String> workerId2Index = new HashMap<String, String>();
	private HashMap<String, String> workerIndex2Id = new HashMap<String, String>();
	
}
