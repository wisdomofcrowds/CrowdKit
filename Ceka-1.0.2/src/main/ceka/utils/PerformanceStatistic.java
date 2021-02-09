/**
 * 
 */
package ceka.utils;

import java.util.ArrayList;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import ceka.core.Dataset;
import ceka.core.Example;


/**
 * Performance Statistic for zero-one loss
 *
 */
public class PerformanceStatistic {

	public PerformanceStatistic() {
	}
	
	/**
	 * get inference accuracy
	 * @return
	 */
	public double getAccuracy() {
		return metrics.accuracy;
	}
	
	/**
	 * get inference accuracy of category k
	 * @param k
	 * @return
	 */
	public double getAccuracyCategory(int k) {
		return rateCorrectCategory[k];
	}
	
	/**
	 * get inference AUC
	 * @return
	 */
	public double getAUC() {
		return metrics.auc;
	}
	
	/**
	 * get inference convex AUC
	 * @return
	 */
	public double getAUCConvex() {
		return metrics.aucConvex;
	}
	
	public double getPresicionBinary() {
		return bcf.tp / (bcf.tp + bcf.fp);
	}
	
	public double NegativePredictiveValue () {
		return bcf.tn / (bcf.tn + bcf.fn);
	}
	
	public double getRecallBinary() {
		return bcf.tp / (bcf.tp + bcf.fn);
	}

	public double getSensitivityBinary() {
		return getRecallBinary();
	}
	
	public double getTruePositiveRateBinary () {
		return getRecallBinary();
	}

	public double getSpecificityBinary() {
		return bcf.tn / (bcf.tn + bcf.fp);
	}
	
	public double getTrueNagativeRateBinary () {
		return getSpecificityBinary();
	}
	
	public double getFMeasureBinary(double beta) {
		return (1 + Math.pow(beta, 2)) * ((getPresicionBinary () * getRecallBinary())/(Math.pow(beta, 2)* getPresicionBinary() + getRecallBinary()));
	}
	
	public double getF1MeasureBinary() {
		return getFMeasureBinary(1);
	}
	
	/**
	 * do statistic
	 * @param dataset
	 */
	public void stat(Dataset dataset)
	{
		int numCategory = dataset.getCategorySize();
		int exampleSize = dataset.getExampleSize();
		
		numExampleCorrectCategory  = new int[numCategory];
		numExampleCategory = new int[numCategory];
		rateCorrectCategory = new double[numCategory];
		
		predictedLabels = new int[exampleSize];
		realLabels = new int[exampleSize];
		
		int correct = 0;		
		for (int i = 0; i < exampleSize; i++)
		{
			Example example = dataset.getExampleByIndex(i);
			
			int integratedLabel = example.getIntegratedLabel().getValue();
			int realLabel = example.getTrueLabel().getValue();
			numExampleCategory[realLabel] += 1;
			if (realLabel == integratedLabel)
			{
				correct += 1;
				numExampleCorrectCategory[realLabel] += 1;
			}
			predictedLabels[i] = integratedLabel;
			realLabels[i] = realLabel;
		}
		
		for (int k = 0; k < numExampleCorrectCategory.length; k++)
			rateCorrectCategory[k] = (double)  numExampleCorrectCategory[k] / (double)  numExampleCategory[k];
		
		metrics.accuracy = (double)correct / (double)exampleSize;
		
		// AUC and AUCMax
		for (int i = 0; i < numCategory - 1; i++)	
		{
			for (int j = i + 1; j < numCategory; j++)
			{
				double auc_ij = calculateAUC(i, j, predictedLabels, realLabels, false);
				metrics.auc += auc_ij;
				double auc_max_ij = calculateAUC(i, j, predictedLabels, realLabels, true);
				metrics.aucConvex += auc_max_ij;
			}
		}
		
		metrics.auc = (2 * metrics.auc) / (double) (numCategory * (numCategory - 1));
		metrics.aucConvex = (2 * metrics.aucConvex) / (double) (numCategory * (numCategory - 1));
		
		// if binary we static binary metric again
		if (numCategory == 2)
			doStatBinary(dataset);
	}
	
	private void doStatBinary(Dataset dataset) {
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example e = dataset.getExampleByIndex(i);
			
			if (e.getIntegratedLabel().getValue() == 0) {
				if (e.getTrueLabel().getValue() == 0) {
					bcf.tn += 1;
				} else {
					bcf.fn += 1;
				}
			} else {
				if (e.getTrueLabel().getValue() == 1) {
					bcf.tp += 1;
				} else {
					bcf.fp += 1;
				}
			}
		}	
	}
	
	private double calculateAUC(int c1, int c2, int [] predictedLabels,  int [] realLabels, boolean convex) {
		
		double auc = 0;
		// find all example with realLabel= C1 && (predictedLabels = C1 or C2);
		ArrayList<Integer> realC1 = new ArrayList<Integer>();
		ArrayList<Double>  predC12 = new ArrayList<Double>();
		
		for (int i = 0; i < realLabels.length; i++) {
			if ((realLabels[i] == c1) && ((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				realC1.add(realLabels[i]);
				predC12.add(new Double(predictedLabels[i]));
			}
			if ((realLabels[i] == c2) && ((predictedLabels[i] == c1) ||  (predictedLabels[i] == c2))) {
				realC1.add(realLabels[i]);
				predC12.add(new Double(predictedLabels[i]));
			}
		}
		
		if (realC1.size() > 0) {
			int [] real = new int[realC1.size()];
			double [] pred = new double [realC1.size()];
			for (int i = 0; i < realC1.size(); i++) {
				if (realC1.get(i).intValue() == c2)
					real[i] = 1;
				else
					real[i] = 0;
				if (Misc.isDoubleSame(predC12.get(i).doubleValue(), (double)c2, 0.0000001))
					pred[i] = 1.0;
				else
					pred[i] = 0.0;
			}
			mloss.roc.Curve rocAnalysis = new mloss.roc.Curve.PrimitivesBuilder().predicteds(pred).actuals(real).build();
			if (convex) {
				// Get the convex hull
				mloss.roc.Curve convexHull = rocAnalysis.convexHull();
			    auc = convexHull.rocArea();
			    if (Double.isNaN(auc))
			    	auc = 0;
			    log.debug("AUC_Convex ("+c1 + "," + c2 +")=" + auc);
			} else {
				auc = rocAnalysis.rocArea();
				if (Double.isNaN(auc))
				    auc = 0;
				log.debug("AUC ("+c1 + "," + c2 +")=" + auc + "    ");
			}
		}
		
		return auc;
	}
	
	private Metrics metrics = new Metrics();
	private int [] numExampleCorrectCategory = null; // number of examples inferred correctly of each category
	private int [] numExampleCategory = null;		 // number of examples of each category
	private double [] rateCorrectCategory = null;	 // correct rate of each category
	private int [] predictedLabels = null;
	private int [] realLabels = null;
	private BinaryConfusionMatrix bcf = new BinaryConfusionMatrix();
	
	private class BinaryConfusionMatrix {
		// Actual =>    0   1 
		// Outcome   0  tn  fn
		// outcome   1  fp  tp
		double tn = 0;
		double tp = 0;
		double fn = 0;
		double fp = 0;
	}
	
	private static Logger log = LogManager.getLogger(PerformanceStatistic.class);
}
