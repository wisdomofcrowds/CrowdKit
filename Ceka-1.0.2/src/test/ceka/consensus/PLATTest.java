/**
 * 
 */
package ceka.consensus;

import java.io.File;
import java.io.FileInputStream;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.ConfigurationSource;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.logging.log4j.LogManager;

import ceka.consensus.plat.PLAT;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.utils.PerformanceStatistic;

/**
 * @author Zhang
 *
 */
public class PLATTest {

	private static Logger log = null;
	
	private static String dataDir = "D:/CekaSpace/Ceka/data/real-world/Processed/BinaryBiased/";
	private static String responseFix = ".response.txt";
	private static String goldFix = ".gold.txt";
	private static String runDir = "D:/CekaSpace/Temp/PLAT/";
	private static String log4jCfgPath = "D:/CekaSpace/Ceka/cfg/log4j2.xml";
	
	public static void main(String[] args) throws Exception {
		
		File config=new File(log4jCfgPath);
		ConfigurationSource source = null;
		source = new ConfigurationSource(new FileInputStream(config),config);
		Configurator.initialize(null, source);
		log = LogManager.getLogger(GTICTest.class);
		
		String dataName = "trec2010";
		
		File testDir  = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();
		
		// load data sets
		String responsePath = dataDir + dataName + responseFix;
		String goldPath = dataDir + dataName + goldFix;
		Dataset dataset = null;
		dataset = FileLoader.loadFile(responsePath, goldPath);
		
		PLAT plat  = new PLAT();
		plat.setUseQuadraticFitting(true);
		plat.doInference(dataset);
		
		PerformanceStatistic reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("PLAT accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() 
				+ " F1:" +  reporter.getF1MeasureBinary());
	}
}
