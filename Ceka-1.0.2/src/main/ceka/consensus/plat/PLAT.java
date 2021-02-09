package ceka.consensus.plat;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import ceka.core.Example;
import ceka.core.Label;

/**
 * Ground truth inference algorithm for binary categorization with labeling bias.
 * Assumption:
 * (1) Negative (class 0) Examples are majority.
 * (2) Positive (class 1) Examples are minority.
 * (3) Quality of labeling on the negative is much greater than that on the positive.<br>
 * <br> Article:<br>
 * Zhang, J., Sheng, V.S., Wu, X. Imbalanced Multiple Noisy Labeling. IEEE Transaction on Knowledge and Data Engineering, 2014.
 * @author Jing Zhang
 *
 */
public class PLAT {
	
	public static final String POSVALUESTR = "1";
	public static final String NEGVALUESTR = "0";
	public static final String NAME = "PLAT";
	
	public PLAT() {
		plat = new PLATCore();
	}
	
	/**
	 * Whether to use Quadratic Fitting when estimating threshold
	 * default: false
	 * @param flag
	 */
	public void setUseQuadraticFitting(boolean flag) {
		plat.setUseQuadraticFitting(flag);
	}
	
	public void doInference(ceka.core.Dataset dataset) {
		plat.buildFreqTable(dataset);
		plat.thresholdMethod();
		for (int i = 0; i < plat.posLabelExamples.size(); i++) {
			Example posE = dataset.getExampleByIndex(plat.posLabelExamples.get(i));
			Label integratedLabel = new Label(null, POSVALUESTR, posE.getId(), NAME);
			posE.setIntegratedLabel(integratedLabel);
		}
		for (int i = 0; i < plat.negLabelExamples.size(); i++) {
			Example negE =  dataset.getExampleByIndex(plat.negLabelExamples.get(i));
			Label integratedLabel = new Label(null, NEGVALUESTR, negE.getId(), NAME);
			negE.setIntegratedLabel(integratedLabel);
		}
		// this is important
		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	public void writeInfo(String infoPath) throws IOException {
		FileWriter fw = new FileWriter(new File(infoPath));
		plat.printInfo(fw);
		fw.close();
	}
	
	PLATCore plat = null;
}
