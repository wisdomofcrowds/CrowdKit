package ceka.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

public class DatasetManipulator {

	/**
	 * split datase to $num sub datasets containing different examples.
	 * @param dataset the dataset is unchanged after being called
	 * @param nFold
	 * @param isShuffled whether randomly shuffle the original dataset 
	 * @return number of $nFold sub datasets
	 */
	public static Dataset [] split(Dataset dataset, int nFold, boolean isShuffled) {
		Dataset [] datasets = new Dataset[nFold];
		int count = dataset.numInstances() / nFold;
		int lastCount = count + dataset.numInstances() % nFold;
		if (isShuffled) {
			Random random = new Random();
			dataset.randomize(random);
		}
		for (int i = 0; i < nFold - 1; i++) {
			datasets[i] = dataset.generateEmpty();
			for (int j = 0; j < count; j++)
				datasets[i].addExample(dataset.getExampleByIndex(i * count + j));
		}
		datasets[nFold - 1]  = dataset.generateEmpty();
		for (int j = 0; j < lastCount; j++) {
			datasets[nFold - 1].addExample(dataset.getExampleByIndex((nFold - 1) * count + j));
		}
		return datasets;
	}
	
	/**
	 * pick up $select dataset from datasets and combine the remaining to a whole one
	 * @param datasets the dataset is unchanged after being called
	 * @param select
	 * @return two datasets, the first is combined one and the second is selected one
	 */
	public static Dataset [] pickCombine(Dataset [] datasets, int select) {
		Dataset [] results = new Dataset[2];
		results[0] = datasets[0].generateEmpty();
		results[1] = datasets[0].generateEmpty();
		for (int i = 0; i < datasets.length; i++)
			if (i != select)
				for (int j = 0; j < datasets[i].numInstances(); j++)
					results[0].addExample(datasets[i].getExampleByIndex(j));
			else
				for (int j = 0; j < datasets[i].numInstances(); j++)
					results[1].addExample(datasets[i].getExampleByIndex(j));
		return results;
	}
	
	/**
	 * add all examples in dataset2 into dataset1
	 * @param ds1 dataset 1
	 * @param ds2 dataset 2
	 */
	public static void addAllExamples(Dataset ds1, Dataset ds2) {
		for (int i = 0; i < ds2.getExampleSize(); i++)
			ds1.addExample(ds2.getExampleByIndex(i));
	}
	
	/**
	 * Split a data set into a training set and a validating set. Instances belonging to each
	 * class in original data set are split at the same ratio, but the instances belonging to 
	 * newly created two data sets are randomly chosen. 
	 * @param dataset original data set
	 * @param cut     the cut point between training set and validation set
	 * @return one training set and one test set 
	 */
	public static Dataset [] splitRandAcrossClass(Dataset dataset, double cut) {
		Dataset [] trainTestDatasets = new Dataset[2];
		
		trainTestDatasets[0] = new Dataset(dataset, 0); // training set
		trainTestDatasets[1] = new Dataset(dataset, 0); // validation set
		
		// see how many classes
		int numCategory = dataset.getCategorySize();
		// copy the categories to two newly created data sets
		for (int i = 0; i < numCategory; i++) {
			Category cate = dataset.getCategory(i);
			trainTestDatasets[0].addCategory(cate.copy());
			trainTestDatasets[1].addCategory(cate.copy());
		}
		// create numCategory example lists
		ArrayList<ArrayList<Example>> exampleLists = new ArrayList<ArrayList<Example>>();
		for (int c = 0; c < numCategory; c++) {
			// process every category
			ArrayList<Example> exampleList = new ArrayList<Example>();
			exampleLists.add(exampleList);
		}
		// sort all examples 
		int numInstance = dataset.numInstances();
		for (int i = 0; i < numInstance; i++) {
			Example example = dataset.getExampleByIndex(i);
			int cate = (int) example.classValue();
			exampleLists.get(cate).add(example);
		}
		
		// split data sets
		for (int c = 0; c < numCategory; c++) {
			ArrayList<List<Example>> cateLists = null;
			cateLists = Misc.splitRandom(exampleLists.get(c), cut);
			
			for (Example e : cateLists.get(0)) {
				Example cpE = (Example)e.copy();
				trainTestDatasets[0].addExample(cpE);
				MultiNoisyLabelSet mnls = cpE.getMultipleNoisyLabelSet(0);
				for (int i = 0; i <  mnls.getLabelSetSize(); i++) {
					Label noisyLabel = mnls.getLabel(i);
					String wId = noisyLabel.getWorkerId();
					Worker worker = trainTestDatasets[0].getWorkerById(wId);
					if (worker == null)
						trainTestDatasets[0].addWorker(worker = new Worker(wId));
					worker.addNoisyLabel(noisyLabel);
				}
			}
			for (Example e : cateLists.get(1)) {
				Example cpE = (Example)e.copy();
				trainTestDatasets[1].addExample(cpE);
				MultiNoisyLabelSet mnls = cpE.getMultipleNoisyLabelSet(0);
				for (int i = 0; i <  mnls.getLabelSetSize(); i++) {
					Label noisyLabel = mnls.getLabel(i);
					String wId = noisyLabel.getWorkerId();
					Worker worker = trainTestDatasets[1].getWorkerById(wId);
					if (worker == null)
						trainTestDatasets[1].addWorker(worker = new Worker(wId));
					worker.addNoisyLabel(noisyLabel);
				}
			} 
		}
		
		return trainTestDatasets;
	}
	
	/**
	 * extract mislabeled instances from the original dataset
	 * @param dataset dataset to be checked
	 * @return a dataset containing all mislabeled instances
	 */
	public static Dataset extractMislabeledData(Dataset dataset) {
		Dataset result = dataset.generateEmpty();
		int exampleSize = dataset.getExampleSize();	
		for (int i = 0; i < exampleSize; i++) {
			Example example = dataset.getExampleByIndex(i);
			int integratedLabel = example.getIntegratedLabel().getValue();
			int realLabel = example.getTrueLabel().getValue();
			if (realLabel != integratedLabel) {
				result.addExample(example);
			}
		}
		return result;
	}
	
	public static Dataset duplicate(Dataset dataset) {
		Dataset cpData = new Dataset(dataset, 0);
		
		// see how many classes
		int numCategory = dataset.getCategorySize();
		// copy the categories to two newly created data sets
		for (int i = 0; i < numCategory; i++) {
			Category cate = dataset.getCategory(i);
			cpData.addCategory(cate.copy());
		}
		
		int numInst = dataset.getExampleSize();
		for (int i = 0; i < numInst; i++) {
			cpData.addExample((Example)dataset.getExampleByIndex(i).copy());
		}
		return cpData;
	}
	
	public static Dataset duplicate(Dataset dataset, ArrayList<Example> except) {
		Dataset cpData = new Dataset(dataset, 0);
		
		// see how many classes
		int numCategory = dataset.getCategorySize();
		// copy the categories to two newly created data sets
		for (int i = 0; i < numCategory; i++) {
			Category cate = dataset.getCategory(i);
			cpData.addCategory(cate.copy());
		}
		
		int numInst = dataset.getExampleSize();
		for (int i = 0; i < numInst; i++) {
			Example e = (Example)dataset.getExampleByIndex(i).copy();
			if (Misc.getElementById(except, e.getId())==null)
				cpData.addExample(e);
		}
		return cpData;
	}
	
	public static void remapDataset(Dataset dataset, String mapDir) throws IOException {
		HashMap<String, Integer> exampleMap = new HashMap<String, Integer>();
		HashMap<String, Integer> workerMap = new HashMap<String, Integer>();
		
		for (int i  = 0; i < dataset.getExampleSize(); i++)
			exampleMap.put(dataset.getExampleByIndex(i).getId(), i);
		for (int j = 0; j < dataset.getWorkerSize(); j++)
			workerMap.put(dataset.getWorkerByIndex(j).getId(), j);
		
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example e = dataset.getExampleByIndex(i);
			MultiNoisyLabelSet mnls = e.getMultipleNoisyLabelSet(0);
			for (int r = 0; r < mnls.getLabelSetSize(); r++) {
				Label noisyLabel = mnls.getLabel(r);
				noisyLabel.setExampleId(exampleMap.get(e.getId()).toString());
				noisyLabel.setWorkerId(workerMap.get(noisyLabel.getWorkerId()).toString());
			}
			if (mnls.getIntegratedLabel() != null)
				mnls.getIntegratedLabel().setExampleId(exampleMap.get(e.getId()).toString());
			if (e.getTrueLabel() != null)
				e.getTrueLabel().setExampleId(exampleMap.get(e.getId()).toString());
			e.setId(exampleMap.get(e.getId()).toString());
		}
		for (int j = 0; j < dataset.getWorkerSize(); j++) {
			Worker w = dataset.getWorkerByIndex(j);
			w.setId(workerMap.get(w.getId()).toString());
		}
		
		if (mapDir != null) {
			File mapDirFile  = new File(mapDir);
			if (!mapDirFile.exists())
				mapDirFile.mkdirs();
			String exampleMapPath = mapDir + dataset.hashCode() + "-EMAP.txt";
			String workerMapPath = mapDir + dataset.hashCode() + "-WMAP.txt";
			FileWriter wFile = new FileWriter(new File(exampleMapPath));
			for (String key : exampleMap.keySet())
				wFile.write(key + "\t\t" + exampleMap.get(key) + "\n");
			wFile.close();
			wFile = new FileWriter(new File(workerMapPath));
			for (String key : workerMap.keySet())
				wFile.write(key + "\t\t" + workerMap.get(key) + "\n");
			wFile.close();
		}
	}
}
