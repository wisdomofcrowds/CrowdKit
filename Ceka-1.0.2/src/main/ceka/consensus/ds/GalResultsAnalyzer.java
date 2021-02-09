/**
 * 
 */
package ceka.consensus.ds;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;

/**
 * @author Zhang
 *
 */
public class GalResultsAnalyzer {
	
	public GalResultsAnalyzer(String methodName) {
		this.methodName = new String(methodName);
	}
	
	public void analyze (String resultPath) throws IOException {
		
		FileReader readerRst = new FileReader(resultPath);
		BufferedReader brRst = new BufferedReader(readerRst);
		String strRst = null;
		
		while((strRst = brRst.readLine()) != null) {
			String [] substrsRst = strRst.split("[ \t]");
			resultMap.put(substrsRst[0], substrsRst[1]);
		}
		
		brRst.close();
		readerRst.close();
	}
	
	/**
	 * assign integrated label to each example in dataset
	 * @param dataset
	 */
	public void assignIntegratedLabel(Dataset dataset) {
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example example = dataset.getExampleByIndex(i);
			String cate = resultMap.get(example.getId());
			Label integratedL = new Label(null, cate, example.getId(), methodName);
			example.setIntegratedLabel(integratedL);
		}
	}

	private String methodName = null;
	private HashMap<String, String> resultMap = new HashMap<String, String>();
}
