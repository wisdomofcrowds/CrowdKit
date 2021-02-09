package ceka.consensus.square;

import java.io.IOException;

import org.square.qa.analysis.Main;

import ceka.converters.FileSaver;
import ceka.core.Dataset;
import ceka.utils.Misc;

/**
 * This class wraps the implementation of two inference algorithms in SQUARE. <br><br>
 * Articles <br>
 * Zen: Gianluca Demartini, Djellel Eddine Difallah, and Philippe Cudr�-Mauroux.
Zencrowd: leveraging probabilistic reasoning and crowdsourcing techniques for large-scale entity
linking. In Proceedings of the 21st international conference on World Wide Web, pages
469�478. ACM, 2012. <br>
  Raykar: Vikas C Raykar, Shipeng Yu, Linda H Zhao, Gerardo Hermosillo Valadez, Charles Florin,
Luca Bogoni, and Linda Moy. Learning from crowds. The Journal of Machine Learning
Research, 11:1297�1322, 2010.
 * @author Zhang
 */

public class SquareIntegration {

	public static String methodMajority = "Majority";
	public static String methodZenCrowd = "Zen";
	public static String methodRYBinary = "Raykar";
	
	public static String estimationTypeUnsupervised = "unsupervised";
	
	/**
	 * create Square Label Integration Object
	 * @param outputDir the outputDIr that the running results are stored
	 */
	public SquareIntegration (String outputDir) {
		if (outputDir.charAt(outputDir.length() - 1) == '/')
			this.saveDir = new String(outputDir + "Square/");
		else if (outputDir.charAt(outputDir.length() - 1) == '\\')
			this.saveDir = new String(outputDir + "Square\\");
		else
			this.saveDir = new String(outputDir + "/Square/");
		Misc.createDirectory(this.saveDir);
		process = new Main();
		analyzer = new SquareResultsAnalyzer(this.saveDir);
	}
	
	/**
	 * infer the integrated label of each example in dataset
	 * @param responsePath
	 * @param goldPath
	 * @param categoryPath
	 */
	public void doInference(Dataset dataset, String methodName) {
		// create temp files
		String relationName = dataset.relationName();
		String responsePath = saveDir + relationName + responseSuffix;
		String goldPath = saveDir +  relationName + goldSuffix;
		String categoryPath = saveDir + relationName + categorySuffix;
		try {
			FileSaver.saveDataset(dataset, responsePath, goldPath, categoryPath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return;
		}
		doInference(responsePath, categoryPath, methodName);
		analyzer.assignIntegratedLabel(dataset);
		// this is important
		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	/**
	 * infer the integrated label of each example in dataset
	 * @param responsePath
	 * @param goldPath
	 * @param categoryPath
	 * @param methodName
	 */
	private void doInference(String responsePath, String categoryPath, String methodName) {
		String [] args = new String[10];
		args[0] = new String("--responses");
		args[1] = new String(responsePath);
		args[2] = new String("--category");
		args[3] = new String(categoryPath);
		args[4] = new String("--saveDir");
		args[5] = new String(saveDir);
		args[6] = new String("--method");
		args[7] = new String(methodName);
		args[8] = new String("--estimation");
		args[9] = new String(estimationTypeUnsupervised);
		process.setupEnvironment(args);
		try {
			process.flow();
			analyzer.analyze(methodName, estimationTypeUnsupervised);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return;
		}
	}
	
	private String saveDir = null;
	private Main process = null;
	private SquareResultsAnalyzer analyzer = null;
	
	private static String responseSuffix = ".response.txt";
	private static String goldSuffix = ".gold.txt";
	private static String categorySuffix = ".category.txt";
}
