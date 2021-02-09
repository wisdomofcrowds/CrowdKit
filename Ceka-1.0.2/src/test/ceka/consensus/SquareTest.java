package ceka.consensus;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.ConfigurationSource;
import org.apache.logging.log4j.core.config.Configurator;

import ceka.consensus.square.SquareIntegration;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.utils.PerformanceStatistic;

public class SquareTest {

	private static Logger log = null;
	
	private static String dataDir = "D:/CekaSpace/Ceka/data/real-world/Processed/GTIC/";
	private static String responseFix = ".response.txt";
	private static String goldFix = ".gold.txt";
	private static String runDir = "D:/CekaSpace/Temp/GTIC/";
	
	private static String log4jCfgPath = "D:/CekaSpace/Ceka/cfg/log4j2.xml";
	
	private static class Result
	{
		long estimatedTime;
		double acc;
		double auc;
		@SuppressWarnings("unused")
		double aucMax;
		ArrayList<Double> cateAcc = new ArrayList<Double>();
		int [] initialCentroids = null;
	}
	
	public static void main(String[] args) throws Exception {
		File config=new File(log4jCfgPath);
		ConfigurationSource source = null;
		source = new ConfigurationSource(new FileInputStream(config),config);
		Configurator.initialize(null, source);
		log = LogManager.getLogger(GTICTest.class);
		
		// ten real-word data sets
		
		//String dataName = "trec2010";
		//String dataName = "adult2";
		//String dataName = "valence5";
		//String dataName = "aircrowd6";
		//String dataName = "valence7";
		//String dataName = "leaves16";
		//String dataName = "fej2013";
		//String dataName = "saj2013";
		//String dataName = "leaves9";
		//String dataName = "aircrowd11";
		String dataName = "synth4";
		
		
		File testDir  = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();
		
		// load data sets
		String responsePath = dataDir + dataName + responseFix;
		String goldPath = dataDir + dataName + goldFix;
		Dataset dataset = null;
		dataset = FileLoader.loadFile(responsePath, goldPath);
		
		// create Square Instance
		SquareIntegration si = new SquareIntegration(runDir);
		
		// run algorithm
		si.doInference(dataset, SquareIntegration.methodZenCrowd);
		
		// get results
		Result rst = new Result();	
		PerformanceStatistic reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		rst.acc = reporter.getAccuracy();
		rst.auc = reporter.getAUC();
		rst.aucMax = reporter.getAUCConvex();
		
		// print results
		for (int k = 0; k < dataset.getCategorySize(); k++)
			rst.cateAcc.add(new Double(reporter.getAccuracyCategory(k)));
		log.info("------------RESULT-----------------------------");
		for (int k = 0; k < dataset.getCategorySize(); k++)
			log.info("Class " + k + " :" +  rst.cateAcc.get(k));
		log.info("Overall Accuracy: " + rst.acc);
		log.info("M-AUC: " +  rst.auc);
		log.info("Running Time: " + rst.estimatedTime);
		//log.info("mAUCMax: " + rst.aucMax);
	}

}
