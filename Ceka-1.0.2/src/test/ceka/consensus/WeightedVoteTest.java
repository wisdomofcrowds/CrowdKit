/**
 * 
 */
package ceka.consensus;

import java.io.File;
import java.io.FileInputStream;

import org.apache.logging.log4j.core.config.Configurator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.ConfigurationSource;

import ceka.consensus.mv.AdaptiveWeightedMajorityVote;
import ceka.consensus.mv.MajorityVote;
import ceka.consensus.plat.PLAT;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.utils.PerformanceStatistic;

/**
 * @author Zhang
 *
 */
public class WeightedVoteTest {
	
	private static Logger log = null;
	
	private static String dataDir = "D:/CekaSpace/Ceka/data/real-world/Processed/BinaryBiased/";
	private static String responseFix = ".response.txt";
	private static String goldFix = ".gold.txt";
	private static String runDir = "D:/CekaSpace/Temp/PLAT/";
	private static String log4jCfgPath = "D:/CekaSpace/Ceka/cfg/log4j2.xml";
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		File config=new File(log4jCfgPath);
		ConfigurationSource source = null;
		source = new ConfigurationSource(new FileInputStream(config),config);
		Configurator.initialize(null, source);
		log = LogManager.getLogger(GTICTest.class);
		
		//String dataName = "anger.median";
		//String dataName = "disgust.median";
		//String dataName = "fear.median";
		//String dataName = "joy.median";
		//String dataName = "sadness.median";
		//String dataName = "surprise.median";
		//String dataName = "valence.median";
		//String dataName = "anger.minentropy";
		//String dataName = "disgust.minentropy";
		//String dataName = "fear.minentropy";
		//String dataName = "joy.minentropy";
		//String dataName = "sadness.minentropy";
		//String dataName = "surprise.minentropy";
		//String dataName = "valence.minentropy";
		//String dataName = "spam";
		
		String dataName = "duck";
		
		File testDir  = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();
		
		// load data sets
		String responsePath = dataDir + dataName + responseFix;
		String goldPath = dataDir + dataName + goldFix;
		Dataset dataset = null;
		dataset = FileLoader.loadFile(responsePath, goldPath);
		
		MajorityVote mv = new MajorityVote();
		mv.doInference(dataset);
		PerformanceStatistic reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("MV accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() + " Precision: " + reporter.getPresicionBinary()
				+ " F1:" +  reporter.getF1MeasureBinary());
		
		PLAT plat = new PLAT();
		plat.doInference(dataset);
		
		reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("PLAT accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() + " Precision: " + reporter.getPresicionBinary()
				+ " F1:" +  reporter.getF1MeasureBinary());
		
		AdaptiveWeightedMajorityVote wv = new AdaptiveWeightedMajorityVote();
		wv.doInference(dataset);
		reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("WMV accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() + " Precision: " + reporter.getPresicionBinary()
				+ " F1:" +  reporter.getF1MeasureBinary());
	}

}
