/**
 * 
 */
package ceka.converters;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.simulation.ExampleMask;
import ceka.simulation.ExampleWorkersMask;
import ceka.utils.Misc;

/**
 * Save Dataset to standard files
 *
 */
public class FileSaver {

	/**
	 * save a dataset to response and gold files
	 * @param dataset cannot be null
	 * @param responsePath
	 * @param goldPath
	 * @throws IOException 
	 */
	public static void saveDataset(Dataset dataset, String responsePath, String goldPath) throws IOException {
		FileWriter responseFile = new FileWriter(new File(responsePath));
		FileWriter goldFile = new FileWriter(new File(goldPath));
		
		boolean firstLineResp = true;
		boolean firstLineGold = true;
		int numExample = dataset.getExampleSize();
		for (int i = 0; i < numExample; i++) {
			Example example = dataset.getExampleByIndex(i);
			ArrayList<String> workerIds = example.getWorkerIdList();
			for (String wId : workerIds) {
				Worker worker = dataset.getWorkerById(wId);
				Label  noisyLabel =  example.getNoisyLabelByWorkerId(wId);
				if (firstLineResp){
					responseFile.write(worker.getId()+"\t"+example.getId()+"\t"+noisyLabel.getValue());
					firstLineResp = false;
				} else {
					responseFile.write("\n"+worker.getId()+"\t"+example.getId()+"\t"+noisyLabel.getValue());
				}
			}
			if (firstLineGold) {
				goldFile.write(example.getId()+"\t"+example.getTrueLabel().getValue());
				firstLineGold = false;
			} else {
				goldFile.write("\n"+example.getId()+"\t"+example.getTrueLabel().getValue());
			}
		}
		
		responseFile.close();
		goldFile.close();
	}
	
	/**
	 * save a dataset to response, gold and category files
	 * @param dataset
	 * @param responsePath
	 * @param goldPath
	 * @param categoryPath
	 * @throws IOException
	 */
	public static void saveDataset(Dataset dataset, String responsePath, String goldPath, String categoryPath) throws IOException {
		FileWriter categoryFile = new FileWriter(new File(categoryPath));
		boolean firstLine = true;
		int numCate = dataset.getCategorySize();
		for (int i = 0; i < numCate; i++) {
			Category cate = dataset.getCategory(i);
			if (firstLine) {
				categoryFile.write(new Integer(cate.getValue()).toString());
				firstLine = false;
			} else {
				categoryFile.write("\n"+cate.getValue());
			}
		}
		categoryFile.close();
		saveDataset(dataset, responsePath, goldPath);
	}
	
	/**
	 * save data to a arff file.
	 * @param dataset dataset to be saved
	 * @param arffxPath arff file path
	 * @throws Exception 
	 */
	public static void saveDatasetArff(Dataset dataset, String arffPath) throws Exception {
		ArffSaver saver = new ArffSaver();
		File outPath = new File(arffPath);
		saver.setFile(outPath);
		saver.setInstances(dataset);
		saver.writeBatch();
	}
	
	/**
	 * save data to a arffx file. The difference between arffx and arff file is that arffx file contains an
	 * ID map list at the end of the file.
	 * @param dataset dataset to be saved
	 * @param arffxPath arffx file path
	 * @throws Exception 
	 */
	public static void saveDatasetArffx(Dataset dataset, String arffxPath) throws Exception {
		
		ArffSaver saver = new ArffSaver();
		File outPath = new File(arffxPath);
		saver.setFile(outPath);
		saver.setInstances(dataset);
		saver.writeBatch();
		
		FileWriter arffxFile = new FileWriter(new File(arffxPath), true);
		
		arffxFile.write("\n\n@ID-MAP\n");
		int numExample = dataset.getExampleSize();
		for (int i = 0; i < numExample; i++) {
			Example example = dataset.getExampleByIndex(i);
			arffxFile.write(example.getId() + "\n");
		}
		
		arffxFile.close();
	}
	
	/**
	 *  Save data to a response file, gold file and arffx file.
	 * @param dataset
	 * @param responsePath
	 * @param goldPath
	 * @param arffxPath
	 * @param mask 
	 * @throws Exception
	 */
	public static void saveDatasetResponseArffx(Dataset dataset, String responsePath, String goldPath, String arffxPath, ExampleWorkersMask mask) throws Exception {
		
		// save to arffx file
		int numExample = dataset.getExampleSize();
		
		if (arffxPath != null) {
			ArffSaver saver = new ArffSaver();
			File outPath = new File(arffxPath);
			saver.setFile(outPath);
			saver.setInstances(dataset);
			saver.writeBatch();
			
			FileWriter arffxFile = new FileWriter(new File(arffxPath), true);
			
			arffxFile.write("\n\n@ID-MAP\n");
			for (int i = 0; i < numExample; i++) {
				Example example = dataset.getExampleByIndex(i);
				arffxFile.write(example.getId() + "\n");
			}
			
			arffxFile.close();
		}
		
		// save to response and gold file
		
		FileWriter responseFile = new FileWriter(new File(responsePath));
		FileWriter goldFile = null; 
		if (goldPath != null)
			goldFile = new FileWriter(new File(goldPath));
		
		boolean firstLineResp = true;
		boolean firstLineGold = true;
		
		for (int i = 0; i < numExample; i++) {
			Example example = dataset.getExampleByIndex(i);
			ArrayList<String> workerIds = example.getWorkerIdList();
			ArrayList<String> workerMask = null;
			if (mask != null) {
				workerMask = mask.getWorkerMask(example.getId());
			}
			boolean exampleHasResponse = false;
			for (String wId : workerIds) {
				Worker worker = dataset.getWorkerById(wId);
				Label  noisyLabel =  example.getNoisyLabelByWorkerId(wId);
				if (firstLineResp){
					if (mask == null) {
						responseFile.write(worker.getId()+"\t"+example.getId()+"\t"+noisyLabel.getValue()); exampleHasResponse = true;
						firstLineResp = false;
					}else {
						if (Misc.getElementEquals(workerMask, worker.getId()) != null) {
							responseFile.write(worker.getId()+"\t"+example.getId()+"\t"+noisyLabel.getValue());  exampleHasResponse = true;
							firstLineResp = false;
						}
					}
				} else {
					if (mask == null) {
						responseFile.write("\n"+worker.getId()+"\t"+example.getId()+"\t"+noisyLabel.getValue()); exampleHasResponse = true;
					}else {
						if (Misc.getElementEquals(workerMask, worker.getId()) != null) {
							responseFile.write("\n"+worker.getId()+"\t"+example.getId()+"\t"+noisyLabel.getValue()); exampleHasResponse = true;
						} 
					}
				}
			}
			if (goldPath != null) {
				if (firstLineGold) {
					if (mask == null) {
						goldFile.write(example.getId()+"\t"+example.getTrueLabel().getValue());
						firstLineGold = false;
					} else {
						if (exampleHasResponse) { goldFile.write(example.getId()+"\t"+example.getTrueLabel().getValue()); firstLineGold = false;}
					}
				} else {
					if (mask == null) {
						goldFile.write("\n"+example.getId()+"\t"+example.getTrueLabel().getValue());
					} else {
						if (exampleHasResponse) goldFile.write("\n"+ example.getId()+"\t"+example.getTrueLabel().getValue());
					}
				}
			}
		}
		
		responseFile.close();
		if (goldPath != null)
			goldFile.close();
	}
	
	/**
	 *  Save data to a response file, gold file and arffx file.
	 * @param dataset
	 * @param responsePath
	 * @param goldPath
	 * @param arffxPath
	 * @param mask
	 * @throws Exception
	 */
	public static void saveDatasetResponseArffx(Dataset dataset, String responsePath, String goldPath,  String arffxPath, ExampleMask mask) throws Exception {
		
		// save to arffx file
		int numExample = dataset.getExampleSize();
		// save to arffx file
		if (arffxPath != null) {
			Dataset arffData = new Dataset(dataset, 0);
			for (int i = 0; i < numExample; i++) {
				Example e = dataset.getExampleByIndex(i);
				if (mask.isActiveExample(e.getId()))
					arffData.add(e);
			}
			ArffSaver saver = new ArffSaver();
			File outPath = new File(arffxPath);
			saver.setFile(outPath);
			saver.setInstances(arffData);
			saver.writeBatch();
			
			FileWriter arffxFile = new FileWriter(new File(arffxPath), true);
			
			arffxFile.write("\n\n@ID-MAP\n");
			for (int i = 0; i < numExample; i++) {
				Example e = dataset.getExampleByIndex(i);
				if (mask.isActiveExample(e.getId()))
					arffxFile.write(e.getId() + "\n");
			}
			
			arffxFile.close();
		}
		
		// save to response and gold file
		FileWriter responseFile = new FileWriter(new File(responsePath));
		FileWriter goldFile = null;
		if (goldPath != null)
			goldFile = new FileWriter(new File(goldPath));
		
		boolean firstLineResp = true;
		boolean firstLineGold = true;
		
		for (int i = 0; i < numExample; i++) {
			Example example = dataset.getExampleByIndex(i);
			if (mask.isActiveExample(example.getId())) {
				int goldCate = example.getTrueLabel().getValue();
				ArrayList<String> workerIds = example.getWorkerIdList();
				for (String wId : workerIds) {
					Worker worker = dataset.getWorkerById(wId);
					Label  noisyLabel =  example.getNoisyLabelByWorkerId(wId);
					if (firstLineResp){
						responseFile.write(worker.getId()+"\t"+example.getId()+"\t"+ noisyLabel.getValue());
						firstLineResp = false;
					} else {
						responseFile.write("\n"+worker.getId()+"\t"+example.getId()+"\t"+  noisyLabel.getValue());
					}
				}
				if (goldPath != null) {
					if (firstLineGold) {
						goldFile.write(example.getId()+"\t"+ goldCate);
						firstLineGold = false;
					} else {
						goldFile.write("\n"+example.getId()+"\t" + goldCate);
					}
				}
			}
		}
		
		responseFile.close();
		if (goldPath != null)
			goldFile.close();
	}
	
	@SuppressWarnings("unused")
	static private Logger log = LogManager.getLogger(FileSaver.class);
}
