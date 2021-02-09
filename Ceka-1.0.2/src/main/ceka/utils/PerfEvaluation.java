package ceka.utils;

import java.util.ArrayList;
import ceka.core.Dataset;
import ceka.core.Example;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

/**
 * Performance Evaluation
 * For multi-class, the auc is calculated according to
 * D. J. Hand and R. J. Till, ¡°A Simple Generalisation of the Area Under the ROC Curve 
 * for Multiple Class Classification Problems,¡± Machine Learning, 45(2):171-186, 2001.
 * @author jzhang
 *
 */
public class PerfEvaluation {

	public PerfEvaluation() {
		
	}
	
	/**
	 * get accuracy
	 * @return
	 */
	public double getAccuracy() {
		return accuracy;
	}
	
	/**
	 * get accuracy of category k
	 * @param k
	 * @return
	 */
	public double getCategoryAccuracy(int k) {
		return rateCorrectCategory[k];
	}
	
	/**
	 * get AUC
	 * @return
	 */
	public double getAUC() {
		return auc;
	}
	
	/**
	 * get precision for binary case
	 * @return
	 */
	public double getPrecision() {
		return cfb.tp / (cfb.tp + cfb.fp);
	}
	
	/**
	 * get negative predictive value (NPV)
	 * @return
	 */
	public double NegativePredictiveValue () {
		return cfb.tn / (cfb.tn + cfb.fn);
	}
	
	/**
	 * get recall for binary case
	 * @return
	 */
	public double getRecall() {
		return cfb.tp / (cfb.tp + cfb.fn);
	}

	/**
	 * get sensitivity for binary case
	 * @return
	 */
	public double getSensitivity() {
		return getRecall();
	}
	
	/**
	 * get true positive rate for binary case
	 * @return
	 */
	public double getTruePositiveRate() {
		return getRecall();
	}

	/**
	 * get specificity for binary case
	 * @return
	 */
	public double getSpecificity() {
		return cfb.tn / (cfb.tn + cfb.fp);
	}
	
	/**
	 * get true negative rate for binary
	 * @return
	 */
	public double getTrueNegativeRate() {
		return getSpecificity();
	}
	
	/**
	 * get f measure for binary case
	 * @param beta
	 * @return
	 */
	public double getFMeasure(double beta) {
		double numerator = (1 + Math.pow(beta, 2)) * getPrecision() * getRecall();
		double denominator = Math.pow(beta, 2)* getPrecision() + getRecall();
		return  numerator / denominator;
	}
	
	/**
	 * get f1-score for binary case
	 * @return
	 */
	public double getF1() {
		return getFMeasure(1);
	}
	
	/**
	 * get AUC calculated by Kboyd
	 * @see https://github.com/kboyd/Roc
	 * @return
	 */
	public double getKboydAUC() {
		if (predictedLabels == null)
			return 0;
		double aucKboyd = 0;
		int numCategory = nbExampleCategory.length;
		// AUC and AUCMax
		for (int i = 0; i < numCategory - 1; i++)
			for (int j = i + 1; j < numCategory; j++)
				aucKboyd  += computeKboydAUC(i, j, predictedLabels, realLabels, false);
		aucKboyd = (2 * aucKboyd) / (double) (numCategory * (numCategory - 1));
		return aucKboyd;
	}
	
	/**
	 * get AUC Convex Hull calculated by Kboyd
	 * @see https://github.com/kboyd/Roc
	 * @return
	 */
	public double getKboydAUCConvexHull() {
		if (predictedLabels == null)
			return 0;
		double aucKboyd = 0;
		int numCategory = nbExampleCategory.length;
		// AUC and AUCMax
		for (int i = 0; i < numCategory - 1; i++)
			for (int j = i + 1; j < numCategory; j++)
				aucKboyd  += computeKboydAUC(i, j, predictedLabels, realLabels, true);
		aucKboyd = (2 * aucKboyd) / (double) (numCategory * (numCategory - 1));
		return aucKboyd;
	}
	
	/**
	 * do statistic
	 * @param dataset
	 */
	public void stat(Dataset dataset) {
		int numCategory = dataset.getCategorySize();
		int exampleSize = dataset.getExampleSize();
		
		nbExampleCorrectCategory  = new int[numCategory];
		nbExampleCategory = new int[numCategory];
		rateCorrectCategory = new double[numCategory];
		
		predictedLabels = new int[exampleSize];
		realLabels = new int[exampleSize];
		
		accuracy = 0;
		auc = 0;
		
		int correct = 0;		
		for (int i = 0; i < exampleSize; i++) {
			Example example = dataset.getExampleByIndex(i);
			
			int integratedLabel = example.getIntegratedLabel().getValue();
			int realLabel = example.getTrueLabel().getValue();
			nbExampleCategory[realLabel] += 1;
			if (realLabel == integratedLabel) {
				correct += 1;
				nbExampleCorrectCategory[realLabel] += 1;
			}
			predictedLabels[i] = integratedLabel;
			realLabels[i] = realLabel;
		}
		
		for (int k = 0; k < nbExampleCorrectCategory.length; k++)
			rateCorrectCategory[k] = (double)  nbExampleCorrectCategory[k] / (double)nbExampleCategory[k];
		
		accuracy = (double)correct / (double)exampleSize;
		
		// AUC
		for (int i = 0; i < numCategory - 1; i++)
			for (int j = i + 1; j < numCategory; j++)
				auc += computeAUC(i, j, predictedLabels, realLabels);
		
		auc = (2 * auc) / (double) (numCategory * (numCategory - 1));
		
		// if binary we re-process binary metric again
		if (numCategory == 2)
			reProcessBinary(dataset);
	}
	
	private void reProcessBinary(Dataset dataset) {
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example e = dataset.getExampleByIndex(i);
			if (e.getIntegratedLabel().getValue() == 0) {
				if (e.getTrueLabel().getValue() == 0) {
					cfb.tn += 1;
				} else {
					cfb.fn += 1;
				}
			} else {
				if (e.getTrueLabel().getValue() == 1) {
					cfb.tp += 1;
				} else {
					cfb.fp += 1;
				}
			}
		}	
	}
	
	private double computeAUC(int c1, int c2, int [] predictedLabels,  int [] realLabels) {
		ArrayList<Integer> reals = new ArrayList<Integer>();
		ArrayList<Double>  preds = new ArrayList<Double>();
		
		for (int i = 0; i < realLabels.length; i++) {
			if (((realLabels[i] == c1) || (realLabels[i] == c2)) && 
				((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				reals.add(realLabels[i]);
				preds.add(new Double(predictedLabels[i]));
			}
		}
		assert(reals.size() == preds.size());
		if (reals.size() == 0)
			return 0;
		
		int [] realsB = new int[reals.size()];
		double [] predsB = new double[preds.size()];
		ArrayList<Prediction> predictions = new ArrayList<Prediction>();
		double [] dist = new double[2];
		
		for (int i = 0; i < realsB.length; i++) {
			if (reals.get(i).intValue() == c2) {
				realsB[i] = 1;
			} else {
				realsB[i] = 0;
			}
			if (Misc.isDoubleSame(preds.get(i).doubleValue(), (double)c2, 0.0000001)) {
				predsB[i] = 1.0;
				dist[0] = 0.0;
				dist[1] = 1.0;
			} else {
				predsB[i] = 0.0;
				dist[0] = 1.0;
				dist[1] = 0.0;
			}
			NominalPrediction pred = new NominalPrediction(realsB[i], dist);
			predictions.add(pred);
		}
		
		ThresholdCurve tc = new ThresholdCurve();
		Instances result = tc.getCurve(predictions);
		double rocArea = ThresholdCurve.getROCArea(result);
		if (Double.isNaN(rocArea))
			rocArea = getROCAreaRevised(result);
		return rocArea;
	}
	
	private double computeKboydAUC(int c1, int c2, int [] predictedLabels,  int [] realLabels, boolean convex) {
		ArrayList<Integer> reals = new ArrayList<Integer>();
		ArrayList<Double>  preds = new ArrayList<Double>();
		
		for (int i = 0; i < realLabels.length; i++) {
			if (((realLabels[i] == c1) || (realLabels[i] == c2)) && 
				((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				reals.add(realLabels[i]);
				preds.add(new Double(predictedLabels[i]));
			}
		}
		assert(reals.size() == preds.size());
		if (reals.size() == 0)
			return 0;
		
		double retVal = 0;
		int [] realVec = new int[reals.size()];
		double [] predVec = new double [preds.size()];
		for (int i = 0; i < reals.size(); i++) {
			if (reals.get(i).intValue() == c2)
				realVec[i] = 1;
			else
				realVec[i] = 0;
			if (Misc.isDoubleSame(preds.get(i).doubleValue(), (double)c2, 0.0000001))
				predVec[i] = 1.0;
			else
				predVec[i] = 0.0;
		}
		mloss.roc.Curve rocAnalysis = new mloss.roc.Curve.PrimitivesBuilder().predicteds(predVec).actuals(realVec).build();
		if (convex) {
			// Get the convex hull
			mloss.roc.Curve convexHull = rocAnalysis.convexHull();
			retVal = convexHull.rocArea();
		    if (Double.isNaN(retVal))
		    	retVal = 0;
		} else {
			retVal = rocAnalysis.rocArea();
			if (Double.isNaN(retVal))
				retVal = 0;
		}
		return retVal;
	}
	
	/**
	 * using this if ThresholdCurve.getROCArea returns NaN
	 * @param tcurve
	 * @return
	 */
	private static double getROCAreaRevised(Instances tcurve) {
		final String RELATION_NAME = "ThresholdCurve";
		final String TRUE_POS_NAME = "True Positives";
		final String FALSE_POS_NAME = "False Positives";
	    final int n = tcurve.numInstances();
	    if (!RELATION_NAME.equals(tcurve.relationName()) || (n == 0)) {
	      return Double.NaN;
	    }
	    final int tpInd = tcurve.attribute(TRUE_POS_NAME).index();
	    final int fpInd = tcurve.attribute(FALSE_POS_NAME).index();
	    final double[] tpVals = tcurve.attributeToDoubleArray(tpInd);
	    final double[] fpVals = tcurve.attributeToDoubleArray(fpInd);

	    double area = 0.0, cumNeg = 0.0;
	    double totalPos = tpVals[0];
	    double totalNeg = fpVals[0];
	    for (int i = 0; i < n; i++) {
	      double cip, cin;
	      if (i < n - 1) {
	        cip = tpVals[i] - tpVals[i + 1];
	        cin = fpVals[i] - fpVals[i + 1];
	      } else {
	        cip = tpVals[n - 1];
	        cin = fpVals[n - 1];
	      }
	      area += cip * (cumNeg + (0.5 * cin));
	      cumNeg += cin;
	    }
	    // add one to prevent dividing zero
	    if (totalNeg == 0)
	    	totalNeg = 1;
	    if (totalPos == 0)
	    	totalPos = 1;
	    area /= (totalNeg * totalPos);
	    
	    return area;
	  }
	
	/**
	 * 
	 * confusion matrix for binary case
	 *
	 */
	private class ConfusionMatrixB {
		// Actual =>    0   1 
		// Outcome   0  tn  fn
		// outcome   1  fp  tp
		double tn = 0;
		double tp = 0;
		double fn = 0;
		double fp = 0;
	}
	
	private ConfusionMatrixB cfb = new ConfusionMatrixB();
	private double accuracy = 0;  // accuracy
	private double auc = 0;       // Area Under ROC
	
	private int [] nbExampleCorrectCategory = null; // number of examples correctly predicted of each category
	private int [] nbExampleCategory = null;		// number of examples of each category
	private double [] rateCorrectCategory = null;	// correct rate of each category
	private int [] predictedLabels = null;
	private int [] realLabels = null;
}
