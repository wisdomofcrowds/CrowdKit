/**
 * 
 */
package ceka.converters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.Misc;


/**
 * Load standard file and form Dataset
 *
 */
public class FileLoader{
	
	/**
	 * Load response file and gold file to form a Dataset
	 * @param responsePath the path of the .response.txt file
	 * @param goldPath the path of the .gold.txt file
	 * @return Dataset the Dataset created from two files
	 * @throws Exception 
	 */
	public static Dataset loadFile(String responsePath, String goldPath) throws Exception {
		String relationName = Misc.exstractFileName(goldPath, false);
		FastVector attInfo = new FastVector();
		int capacity = 0;
		Dataset dataset = new Dataset(relationName, attInfo, capacity);
		
		ArrayList<Integer> categories = new ArrayList<Integer>();
		
		// read gold file
		FileReader reader = new FileReader(goldPath);
		BufferedReader readerBuffer = new BufferedReader(reader);
		String line = null;
		
		while((line = readerBuffer.readLine()) != null) {
			String [] subStrs = line.split("[ \t]");
			String exampleId = subStrs[0];
			Example example = null;
			if ((example = dataset.getExampleById(exampleId)) != null) {
				log.info("Example <" + exampleId + "> already exists, overwrite the orginal true label");
				Label trueLabel = new Label(null, subStrs[1], exampleId, Worker.WORKERID_GOLD);
				example.setTrueLabel(trueLabel);
			}
			else {
				example = new Example(1, exampleId);
				Label trueLabel = new Label(null, subStrs[1], exampleId, Worker.WORKERID_GOLD);
				example.setTrueLabel(trueLabel);
				dataset.addExample(example);
			}
			Misc.addElementIfNotExistedEquals(categories, Integer.parseInt(subStrs[1]));
		}
		readerBuffer.close();
		reader.close();
		
		// check and categories
		Collections.sort(categories);
		boolean correct = true;
		for (int k = 0; k < categories.size(); k++) {
			if (categories.get(k).intValue() != k) {
				correct = false;
				break;
			}
			Category category = new Category(null, categories.get(k).toString());
			dataset.addCategory(category);
		}
		if (!correct)
			throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		
		// read response file
		reader = new FileReader(responsePath);
		readerBuffer = new BufferedReader(reader);
		line = null;
		
		while((line = readerBuffer.readLine()) != null) {
			String [] subStrs = line.split("[ \t]");
			String workerId = subStrs[0];
			String exampleId = subStrs[1];
			String response = subStrs[2];
			if (subStrs.length > 3) {
				// TODO: multi-label
			}
			// check response
			if (Misc.getElementEquals(categories, Integer.parseInt(response)) == null) {
				readerBuffer.close();
				reader.close();
				throw new Exception("Invalid cateory:" + Integer.parseInt(response)+", which is not found in gold file.");
			}
			Example example = null;
			if ((example = dataset.getExampleById(exampleId)) == null) {
				log.warn("cannot find example <" + exampleId + ">, discards this response.");
			} else {
				Label noisyLabel = new Label(null, response, example.getId(), workerId);
				Worker worker = dataset.getWorkerById(workerId);
				if (worker == null)
					dataset.addWorker(worker = new Worker(workerId));
				worker.addNoisyLabel(noisyLabel);
				example.addNoisyLabel(noisyLabel);
			}
		}
		readerBuffer.close();
		reader.close();
		
		// add a class attribute
		ArrayList<String> cateStrs  = new ArrayList<String>();
		for (Integer cate: categories)
			cateStrs.add(new String(cate.toString()));
		
		Attribute attr = new Attribute("Class", cateStrs);
		dataset.insertAttributeAt(attr, 0);
		dataset.setClassIndex(0);
		
		return dataset;
	}
	
	/**
	 * Load Arff File to form a Dataset
	 * @param arffPath
	 * @return Dataset
	 * @throws Exception 
	 */
	public static Dataset loadFile(String arffPath) throws Exception {
		FileReader reader = new FileReader(arffPath);
		BufferedReader readerBuffer = new BufferedReader(reader);
		Instances instSet = new Instances(readerBuffer);
		// find class attribute
		ArrayList<Integer> categories = new ArrayList<Integer>();
		int numAttrib = instSet.numAttributes();
		for (int i = 0; i < numAttrib; i++) {
			Attribute attr  = instSet.attribute(i);
			String attribName = attr.name();
			if (attribName.equalsIgnoreCase("class")) {
				int numV = attr.numValues();
				for (int j = 0; j < numV; j++) {
					String vStr = attr.value(j);
					Integer vInt = Integer.parseInt(vStr);
					categories.add(vInt);
				}
				instSet.setClassIndex(i);
				break;
			}
		}
		// after weka Instance Set has been created, we can create ourself's data set
		Dataset  dataset = new Dataset(instSet, 0);
		Collections.sort(categories);
		boolean correct = true;
		for (int k = 0; k < categories.size(); k++) {
			if (categories.get(k).intValue() != k) {
				correct = false;
				break;
			}
			Category category = new Category(null, categories.get(k).toString());
			dataset.addCategory(category);
		}
		if (!correct)
			throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		// Create Examples
		
		int numInst =  instSet.numInstances();
		for (int i = 0; i < numInst; i++) {
			Example example = new Example(instSet.instance(i), new Integer(i).toString());
			Integer classValue = new Integer((int)instSet.instance(i).classValue());
			Label trueLabel = new Label(null, classValue.toString(), example.getId(),  Worker.WORKERID_GOLD);
			example.setTrueLabel(trueLabel);
			dataset.addExample(example);
		}
		
		reader.close();
		readerBuffer.close();
		return dataset;
	}
	
	/**
	 * Load responseFile, goldFile, and arffxFile to form Dataset, if goldFile is null, use the class
	 * values in arffixFile as the true label
	 * @param responsePath
	 * @param goldPath
	 * @param arffxPath
	 * @return
	 * @throws Exception
	 */
	public static Dataset loadFileX(String responsePath, String goldPath, String arffxPath) throws Exception {
		String extensionName = Misc.extractFileSuffix(arffxPath);
		if (!extensionName.equalsIgnoreCase("arffx"))
			throw new IllegalArgumentException("should be .arffx file");
		
		// read arffx file
		int index = 0;
		HashMap<Integer, String> idMap = new HashMap<Integer, String>();
		FileReader arffxReader = new FileReader(arffxPath);
		BufferedReader arffxReaderBuffer = new BufferedReader(arffxReader);
		String line = null;
		ArrayList<String> arffContents = new ArrayList<String>();
		boolean beginMap = false;
		while((line = arffxReaderBuffer.readLine()) != null) {
			
			if ((line.length() >= "@ID-MAP".length()) && 
					line.substring(0, "@ID-MAP".length()).equalsIgnoreCase("@ID-MAP")) {
				beginMap = true;
			} else {
				if ((!beginMap) && (line.length() > 0)) {
					arffContents.add(line);
				}else {
					if (line.length() > 0) {
						line = line.trim();
						idMap.put(index, line);
						index++;
					}
				}
			}
		}
		arffxReaderBuffer.close();
		arffxReader.close();
		
		// write contents to arff file
		String fileName = Misc.exstractFileName(arffxPath, false);
		String arffDir = Misc.extractDir(arffxPath);
		String arffPath = arffDir + fileName + ".arff";
		FileWriter arffFile = new FileWriter(new File(arffPath), false);
		for (int i = 0; i < arffContents.size(); i++) {
			arffFile.write(arffContents.get(i) + "\n");
		}
		arffFile.close();
		
		// read arff file
		FileReader reader = new FileReader(arffPath);
		BufferedReader readerBuffer = new BufferedReader(reader);
		Instances instSet = new Instances(readerBuffer);
		// find class attribute
		ArrayList<Integer> categories = new ArrayList<Integer>();
		int numAttrib = instSet.numAttributes();
		for (int i = 0; i < numAttrib; i++) {
			Attribute attr  = instSet.attribute(i);
			String attribName = attr.name();
			if (attribName.equalsIgnoreCase("class")) {
				int numV = attr.numValues();
				for (int j = 0; j < numV; j++) {
					String vStr = attr.value(j);
					Integer vInt = Integer.parseInt(vStr);
					categories.add(vInt);
				}
				instSet.setClassIndex(i);
				break;
			}
		}
		// after weka Instance Set has been created, we can create ourself's data set
		Dataset  dataset = new Dataset(instSet, 0);
		Collections.sort(categories);
		boolean correct = true;
		for (int k = 0; k < categories.size(); k++) {
			if (categories.get(k).intValue() != k) {
				correct = false;
				break;
			}
			Category category = new Category(null, categories.get(k).toString());
			dataset.addCategory(category);
		}
		if (!correct)
			throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		
		// Create Examples
		int numInst =  instSet.numInstances();
		for (int i = 0; i < numInst; i++) {
			// set id in ID-MAP
			Example example = new Example(instSet.instance(i), idMap.get(i));
			Integer classValue = new Integer((int)instSet.instance(i).classValue());
			Label trueLabel = new Label(null, classValue.toString(), example.getId(),  Worker.WORKERID_GOLD);
			example.setTrueLabel(trueLabel);
			dataset.addExample(example);
		}
		
		reader.close();
		readerBuffer.close();
		
		// delete temp arff file
		File arfffile = new File (arffPath);
		arfffile.delete();
		
		// read gold file
		if (goldPath != null) {
			FileReader readerGold = new FileReader(goldPath);
			BufferedReader readerBufferGold = new BufferedReader(readerGold);
			String lineGold = null;
			
			while((lineGold = readerBufferGold.readLine()) != null) {
				String [] subStrs = lineGold.split("[ \t]");
				String exampleId = subStrs[0];
				Example example = null;
				if ((example = dataset.getExampleById(exampleId)) != null) {
					log.debug("Example <" + exampleId + "> already exists, overwrite the orginal true label");
					Label trueLabel = new Label(null, subStrs[1], exampleId, Worker.WORKERID_GOLD);
					example.setTrueLabel(trueLabel);
				}
				else {
					log.warn("Example <" + exampleId + "> exists in gold file but doesn't exist in arff file");
				}
				Misc.addElementIfNotExistedEquals(categories, Integer.parseInt(subStrs[1]));
			}
			readerBufferGold.close();
			readerGold.close();
			// check and categories
			Collections.sort(categories);
			boolean correctGold = true;
			for (int k = 0; k < categories.size(); k++) {
				if (categories.get(k).intValue() != k) {
					correctGold = false;
					break;
				}
				Category category = new Category(null, categories.get(k).toString());
				dataset.addCategory(category);
			}
			if (!correctGold)
				throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		}
		
		// read response file
		reader = new FileReader(responsePath);
		readerBuffer = new BufferedReader(reader);
		line = null;
		
		while((line = readerBuffer.readLine()) != null) {
			String [] subStrs = line.split("[ \t]");
			String workerId = subStrs[0];
			String exampleId = subStrs[1];
			String response = subStrs[2];
			if (subStrs.length > 3) {
				// TODO: multi-label
			}
			// check response
			if (Misc.getElementEquals(categories, Integer.parseInt(response)) == null) {
				readerBuffer.close();
				reader.close();
				throw new Exception("Invalid cateory:" + Integer.parseInt(response)+", which is not found in gold file.");
			}
			Example example = null;
			if ((example = dataset.getExampleById(exampleId)) == null) {
				log.warn("cannot find example <" + exampleId + ">, discards this response.");
			} else {
				Label noisyLabel = new Label(null, response, example.getId(), workerId);
				Worker worker = dataset.getWorkerById(workerId);
				if (worker == null)
					dataset.addWorker(worker = new Worker(workerId));
				worker.addNoisyLabel(noisyLabel);
				example.addNoisyLabel(noisyLabel);
			}
		}
		readerBuffer.close();
		reader.close();
		
		return dataset;
	}
	
	/**
	 * Load responseFile, goldFile, and arffxFile to form Dataset, if goldFile is null, use the class
	 * values in arffiFile as the true label
	 * @param responsePath
	 * @param goldPath
	 * @param arffPath
	 * @return
	 * @throws Exception
	 */
	public static Dataset loadFile(String responsePath, String goldPath, String arffPath) throws Exception {
		String extensionName = Misc.extractFileSuffix(arffPath);
		if (!extensionName.equalsIgnoreCase("arff"))
			throw new IllegalArgumentException("should be .arff file");
		
		// read arff file
		FileReader reader = new FileReader(arffPath);
		BufferedReader readerBuffer = new BufferedReader(reader);
		Instances instSet = new Instances(readerBuffer);
		// find class attribute
		ArrayList<Integer> categories = new ArrayList<Integer>();
		int numAttrib = instSet.numAttributes();
		for (int i = 0; i < numAttrib; i++) {
			Attribute attr  = instSet.attribute(i);
			String attribName = attr.name();
			if (attribName.equalsIgnoreCase("class")) {
				int numV = attr.numValues();
				for (int j = 0; j < numV; j++) {
					String vStr = attr.value(j);
					Integer vInt = Integer.parseInt(vStr);
					categories.add(vInt);
				}
				instSet.setClassIndex(i);
				break;
			}
		}
		// after weka Instance Set has been created, we can create ourself's data set
		Dataset  dataset = new Dataset(instSet, 0);
		Collections.sort(categories);
		boolean correct = true;
		for (int k = 0; k < categories.size(); k++) {
			if (categories.get(k).intValue() != k) {
				correct = false;
				break;
			}
			Category category = new Category(null, categories.get(k).toString());
			dataset.addCategory(category);
		}
		if (!correct)
			throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		
		// Create Examples
		int numInst =  instSet.numInstances();
		for (int i = 0; i < numInst; i++) {
			// set id in ID-MAP
			Example example = new Example(instSet.instance(i), new Integer(i).toString());
			Integer classValue = new Integer((int)instSet.instance(i).classValue());
			Label trueLabel = new Label(null, classValue.toString(), example.getId(),  Worker.WORKERID_GOLD);
			example.setTrueLabel(trueLabel);
			dataset.addExample(example);
		}
		
		reader.close();
		readerBuffer.close();
		
		// read gold file
		if (goldPath != null) {
			FileReader readerGold = new FileReader(goldPath);
			BufferedReader readerBufferGold = new BufferedReader(readerGold);
			String lineGold = null;
			
			while((lineGold = readerBufferGold.readLine()) != null) {
				String [] subStrs = lineGold.split("[ \t]");
				String exampleId = subStrs[0];
				Example example = null;
				if ((example = dataset.getExampleById(exampleId)) != null) {
					log.debug("Example <" + exampleId + "> already exists, overwrite the orginal true label");
					Label trueLabel = new Label(null, subStrs[1], exampleId, Worker.WORKERID_GOLD);
					example.setTrueLabel(trueLabel);
				}
				else {
					log.warn("Example <" + exampleId + "> exists in gold file but doesn't exist in arff file");
				}
				Misc.addElementIfNotExistedEquals(categories, Integer.parseInt(subStrs[1]));
			}
			readerBufferGold.close();
			readerGold.close();
			// check and categories
			Collections.sort(categories);
			boolean correctGold = true;
			for (int k = 0; k < categories.size(); k++) {
				if (categories.get(k).intValue() != k) {
					correctGold = false;
					break;
				}
				Category category = new Category(null, categories.get(k).toString());
				dataset.addCategory(category);
			}
			if (!correctGold)
				throw new Exception("Invalid cateories, categories must be consecutive integers staring from 0");
		}
		
		// read response file
		reader = new FileReader(responsePath);
		readerBuffer = new BufferedReader(reader);
		String line = null;
		
		while((line = readerBuffer.readLine()) != null) {
			String [] subStrs = line.split("[ \t]");
			String workerId = subStrs[0];
			String exampleId = subStrs[1];
			String response = subStrs[2];
			if (subStrs.length > 3) {
				// TODO: multi-label
			}
			// check response
			if (Misc.getElementEquals(categories, Integer.parseInt(response)) == null) {
				readerBuffer.close();
				reader.close();
				throw new Exception("Invalid cateory:" + Integer.parseInt(response)+", which is not found in gold file.");
			}
			Example example = null;
			if ((example = dataset.getExampleById(exampleId)) == null) {
				log.warn("cannot find example <" + exampleId + ">, discards this response.");
			} else {
				Label noisyLabel = new Label(null, response, example.getId(), workerId);
				Worker worker = dataset.getWorkerById(workerId);
				if (worker == null)
					dataset.addWorker(worker = new Worker(workerId));
				worker.addNoisyLabel(noisyLabel);
				example.addNoisyLabel(noisyLabel);
			}
		}
		readerBuffer.close();
		reader.close();
		
		return dataset;
	}
	
	static private Logger log = LogManager.getLogger(FileLoader.class);
}
