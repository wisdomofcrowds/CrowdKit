/**
 * 
 */
package ceka.consensus.ds;

import java.io.IOException;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import com.ipeirotis.gal.engine.Engine;
import com.ipeirotis.gal.engine.EngineContext;

import ceka.converters.FileSaver;
import ceka.core.Dataset;
import ceka.utils.Misc;

/**
 * Article:<br>
 * Alexander Philip Dawid and Allan M Skene. Maximum likelihood estimation of observer
error-rates using the em algorithm. Applied statistics, pages 20 - 28, 1979.
 * @author Zhang
 */
public class GalDawidSkene {
	
	public static String NAME = "GALDS";
	
	public GalDawidSkene (String outputDir) {
		this.saveDir = new String(outputDir);
		Misc.createDirectory(this.saveDir);
		analyzer = new GalResultsAnalyzer(NAME);
	}
	
	/**
	 * infer the integrated label of each example in dataset
	 * @param responsePath
	 * @param goldPath
	 * @param categoryPath
	 */
	public void doInference(Dataset dataset) {
		// create temp files
		String relationName = dataset.relationName();
		String responsePath = saveDir + relationName + responseSuffix;
		String goldPath = saveDir +  relationName + goldSuffix;
		String categoryPath = saveDir + relationName + categorySuffix;
		String resultPath = saveDir +  relationName + resultSuffix;
		try {
			FileSaver.saveDataset(dataset, responsePath, goldPath, categoryPath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return;
		}
		doInference(responsePath, categoryPath, resultPath);
		analyzer.assignIntegratedLabel(dataset);
	}
	
	/**
	 * infer the integrated label of each example in dataset
	 * @param responsePath
	 * @param goldPath
	 * @param categoryPath
	 * @param methodName
	 */
	public void doInference(String responsePath, String categoryPath, String resultPath) {
		String [] args = new String[7];
		args[0] = new String("--categories");
		args[1] = new String(categoryPath);
		args[2] = new String("--input");
		args[3] = new String(responsePath);
		args[4] = new String("--result");
		args[5] = new String(resultPath);
		args[6] = new String("--verbose");
		
		EngineContext ctx = new EngineContext();
		CmdLineParser parser = new CmdLineParser(ctx);
		try {
			parser.parseArgument(args);
			Engine engine = new Engine(ctx);
			engine.execute();
			analyzer.analyze(resultPath);
		} catch (CmdLineException | IOException e) {
			System.err.println(e);
			return;
		}
	}
	
	private String saveDir = null;
	private GalResultsAnalyzer analyzer = null;
	private static String responseSuffix = ".response.txt";
	private static String goldSuffix = ".gold.txt";
	private static String categorySuffix = ".category.txt";
	private static String resultSuffix = ".result.txt";
}
