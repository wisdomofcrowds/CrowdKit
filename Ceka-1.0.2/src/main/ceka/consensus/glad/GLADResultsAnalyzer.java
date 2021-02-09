/**
 * 
 */
package ceka.consensus.glad;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import ceka.converters.GLADConverter;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;

/**
 * @author Zhang
 *
 */
public class GLADResultsAnalyzer {
	
	public GLADResultsAnalyzer(String methodName, double thrs) {
		this.methodName = new String(methodName);
		threshold = thrs;
	}
	
	public void analyze (String resultPath, GLADConverter converter) throws IOException {
		
		FileReader readerRst = new FileReader(resultPath);
		BufferedReader brRst = new BufferedReader(readerRst);
		String strRst = null;
		
		while((strRst = brRst.readLine()) != null) {
			String [] substrsRst = strRst.split("[ \t]");
			Double value = Double.parseDouble(substrsRst[1]);
			if (value.doubleValue() > threshold)
				resultMap.put(converter.getExampleId(substrsRst[0]), "1");
			else
				resultMap.put(converter.getExampleId(substrsRst[0]), "0");
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
	private double threshold = 0.5;
	private HashMap<String, String> resultMap = new HashMap<String, String>();
}
