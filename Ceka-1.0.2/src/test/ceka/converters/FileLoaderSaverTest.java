package ceka.converters;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

public class FileLoaderSaverTest {

	public static void main(String[] args) {
		
		try {
			String responsePath = "D:/CekaSpace/Ceka/data/real-world/income94crowd/income94.response.txt";
			String goldPath = "D:/CekaSpace/Ceka/data/real-world/income94crowd/income94.gold.txt";
			String arffPath = "D:/CekaSpace/Ceka/data/real-world/income94crowd/arff/Income94HighAccGold.arff";
			
			Dataset dataset = FileLoader.loadFile(responsePath, goldPath, arffPath);
			
			for (int i = 0; i < dataset.getWorkerSize(); i++) {
				Worker w = dataset.getWorkerByIndex(i);
			}
			
			Example example = dataset.getExampleById("id");
			MultiNoisyLabelSet mnls = example.getMultipleNoisyLabelSet(0);
			for (int i = 0; i < mnls.getLabelSetSize(); i++) {
				Label label = mnls.getLabel(i);
			}
			
			System.out.println("number of example:" + dataset.getExampleSize());
			
			
			//FileSaver.saveDataset(dataset, saveResponsePath, saveGoldPath);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	private static String responsePath = "D:/zcrom/output/income94.response.txt";;
	private static String goldPath = "D:/zcrom/output/income94.gold.txt";
	
	private static String saveResponsePath = "D:/zcrom/Ceka/data/income94.response.txt";;
	private static String saveGoldPath = "D:/zcrom/Ceka/data/income94.gold.txt";

}
