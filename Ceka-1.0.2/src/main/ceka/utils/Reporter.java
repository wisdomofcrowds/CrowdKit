package ceka.utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.inference.TTest;

public class Reporter {

	public class MethodMetrics implements IdDecorated {
		
		public MethodMetrics (String method) {
			methodName = new String(method);
		}
		
		/* (non-Javadoc)
		 * @see ceka.utils.IdDecorated#getId()
		 */
		@Override
		public String getId() {
			return methodName;
		}
		
		public void setMaxRound(int num) {
			metricsTable.clear();
			for (int i = 0; i < num; i++) {
				metricsTable.add(new ArrayList<Metrics>());
			}
		}
		
		public void write(OutputStreamWriter osw, int round) throws IOException {
			osw.write("***********************************************************\r\n");
			osw.write("                       " + methodName + "(" + round +")\r\n");
			osw.write("***********************************************************\r\n");
			ArrayList<Metrics> metrics = metricsTable.get(round);
			writeAcc(osw, metrics);
			writeAUC(osw, metrics);
		}
		
		public void writeMean(OutputStreamWriter osw) throws IOException {
			osw.write("***********************************************************\r\n");
			osw.write("                       " + methodName + "(mean) \r\n");
			osw.write("***********************************************************\r\n");
			writeAcc(osw, meanMetrics);
			writeAUC(osw, meanMetrics);
		}
		
		public void writeSd(OutputStreamWriter osw) throws IOException {
			osw.write("***********************************************************\r\n");
			osw.write("                       " + methodName + "(sd) \r\n");
			osw.write("***********************************************************\r\n");
			writeAcc(osw, sdMetrics);
			writeAUC(osw, sdMetrics);
		}
		
		public void calculateAverageStdDevaition() {
			int maxRound = metricsTable.size();
			int listLength = metricsTable.get(0).size();
			// calculate mean values
			for (int i = 0; i < listLength; i++) {
				Metrics metrics = new Metrics();
				for (int j = 0; j < maxRound; j++) {
					ArrayList<Metrics> currentlist = metricsTable.get(j);
					metrics.addValues(currentlist.get(i));
				}
				metrics.dividedBy(maxRound);
				meanMetrics.add(metrics);
			}
			// calculate std deviations
			for (int i = 0; i < listLength; i++) {
				Metrics metrics = new Metrics();
				for (int j = 0; j < maxRound; j++) {
					ArrayList<Metrics> currentlist = metricsTable.get(j);
					Metrics mtr = new Metrics(currentlist.get(i));
					mtr.minusValues(meanMetrics.get(i));
					mtr.pow(2);
					metrics.addValues(mtr);
				}
				metrics.dividedBy(maxRound).pow(0.5);
				sdMetrics.add(metrics);
			}
		}
		
		private void writeAcc(OutputStreamWriter osw, ArrayList<Metrics> metricesList) throws IOException{
			osw.write("ACC: ");
			for(int i  = 0; i < metricesList.size(); i++) {
				Metrics metrics = metricesList.get(i);
				String str = String.format("%.4f,", metrics.accuracy);
				osw.write(str);
			}
			osw.write("\r\n");
		}
		
		private void writeAUC(OutputStreamWriter osw, ArrayList<Metrics> metricesList) throws IOException{
			osw.write("AUC: ");
			for(int i  = 0; i < metricesList.size(); i++) {
				Metrics metrics = metricesList.get(i);
				String str = String.format("%.4f,", metrics.auc);
				osw.write(str);
			}
			osw.write("\r\n");
		}
		
		public String methodName = null;
		public ArrayList<ArrayList<Metrics>> metricsTable = new ArrayList<ArrayList<Metrics>>();
		public ArrayList<Metrics> meanMetrics = new ArrayList<Metrics>();  // mean
		public ArrayList<Metrics> sdMetrics = new ArrayList<Metrics>();    // stadard deviation
	}
	
	public class StudentTTest implements IdDecorated {
		
		public static final String METRIC_ACC = "acc";
		public static final String METRIC_AUC = "auc";

		public StudentTTest(String method1, String method2, double pValue) {
			methodA = new String(method1);
			methodB = new String(method2);
			id = methodA + methodB;
			p_value = pValue;
		}
		
		@Override
		public String getId() {
			return id;
		}
		
		public String getMethodA() {
			return methodA;
		}
		
		public String getMethodB() {
			return methodB;
		}
		
		public void doTest(ArrayList<ArrayList<Metrics>> perform1, ArrayList<ArrayList<Metrics>> perform2, String metricName) {
			assert((perform1.size() == perform2.size()) && (perform1.get(0).size() == perform2.get(0).size()));
			int maxRound = perform1.size();
			int number = perform1.get(0).size();
			
			String [] results = new String[number];
			double [] resultValues = new double[number];
			
			for(int i = 0; i < number; i++) {
				double [] perform1Metric = new double[maxRound];
				double [] perform2Metric = new double[maxRound];
				for (int j =0; j < maxRound; j++) {
					if (metricName.equalsIgnoreCase(METRIC_ACC)) {
						perform1Metric[j] = perform1.get(j).get(i).accuracy;
						perform2Metric[j] = perform2.get(j).get(i).accuracy;
					}
					if (metricName.equalsIgnoreCase(METRIC_AUC)) {
						perform1Metric[j] = perform1.get(j).get(i).auc;
						perform2Metric[j] = perform2.get(j).get(i).auc;
					}
				}
				double avgP1= new Mean().evaluate(perform1Metric, 0, perform1Metric.length);
				double avgP2= new Mean().evaluate(perform2Metric, 0, perform2Metric.length);
				if (avgP1 == avgP2)
					results[i] = "t";
				else if (avgP1 > avgP2)
					results[i] = "w";
				else
					results[i] = "l";
				
				TTest test = new TTest();
				resultValues[i] = test.pairedTTest(perform1Metric, perform2Metric);
				if (resultValues[i] > p_value)
					results[i] = "t";
			}
			if (metricName.equalsIgnoreCase(METRIC_ACC))
				accResults = results;
			if (metricName.equalsIgnoreCase(METRIC_AUC))
				aucResults = results;
		}
		
		public void write(OutputStreamWriter osw, String metricName) throws IOException {
			if (metricName.equalsIgnoreCase(METRIC_ACC) && (accResults != null)) {
				for (int i = 0; i < accResults.length; i++)
					osw.write( accResults[i] + ", ");
			}
			if (metricName.equalsIgnoreCase(METRIC_AUC) && (aucResults != null)) {
				for (int i = 0; i < aucResults.length; i++)
					osw.write( aucResults[i] + ", ");
			}
		}
		private String id = null;
		private String methodA = null;
		private String methodB = null;
		private double p_value = 0.05;
		private String [] accResults = null;
		private String [] aucResults = null;
	}
	
	public Reporter(String name) {
		this.name = name;
	}
	
	public void setOutDir(String outDir) {
		File outDirFile  = new File(outDir);
		if (!outDirFile.exists())
			outDirFile.mkdirs();
		this.outDir = outDir;
	}
	
	public void setUserParameterString (String str) {
		useParameterString = new String (str);
	}
	
	public void addMethod(String methodName) {
		MethodMetrics mm = Misc.getElementById(methodList, methodName);
		if (mm == null)
			methodList.add(new MethodMetrics(methodName));
	}
	
	public void setMaxRound(int num) {
		maxRound = num;
		for (MethodMetrics mm : methodList) {
			mm.setMaxRound(maxRound);
		}
	}
	
	public void initialize() {
		instantFilePathBuffer = new StringBuffer();
		instantFilePathBuffer.append(outDir+name+useParameterString+"-");
		Date date = new Date();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd'-'HH'h'mm'm'ss's'");
		String dateStr = sdf.format(date);
		instantFilePathBuffer.append(dateStr + ".record.txt");
		try {
			instantWriter = new OutputStreamWriter(new FileOutputStream(new File(instantFilePathBuffer.toString())));
			instantWriter.write("Method" + ", " + "Round No." + "," + "ACC" + ", " + "AUC"+ "\r\n");
			instantWriter.flush();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void addMetrics(String methodName, int round, Metrics metrics) {
		MethodMetrics mm = Misc.getElementById(methodList, methodName);
		mm.metricsTable.get(round).add(metrics);
		writeInstantFile(methodName, round, metrics);
	}
	
	public void printResults(String outPath) {
		StringBuffer outPathBuffer = new StringBuffer();
		if (outPath == null) {
			outPathBuffer.append(outDir+name+useParameterString+"-");
			Date date = new Date();
			SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd'-'HH'h'mm'm'ss's'");
			String dateStr = sdf.format(date);
			outPathBuffer.append(dateStr + ".results.txt");
			
		} else {
			outPathBuffer.append(outPath);
		}
		
		try {
			FileOutputStream fos = new FileOutputStream(new File(outPathBuffer.toString()));
			streamOut(fos);
			fos.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void streamOut(OutputStream outStream) throws IOException {
		OutputStreamWriter osw = new OutputStreamWriter(outStream);
		osw.write("------------------------------------------------------------\r\n");
		Date date = new Date();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd't'HH'h'mm'm'ss's'");
		String dataStr = sdf.format(date);
		osw.write(name + "  " + dataStr +"\r\n");
		osw.write("Parameters: " + useParameterString + "\r\n");
		osw.write("Total Round Number: " + maxRound + "\r\n");
		osw.write("------------------------------------------------------------\r\n");
		for (int r = 0; r < maxRound; r++) {
			osw.write("-----------------ROUND-" + r +"---------------------------------------\r\n");
			for (int i = 0; i < methodList.size(); i++)
				methodList.get(i).write(osw, r);
		}
		for (int i = 0; i < methodList.size(); i++)
			methodList.get(i).calculateAverageStdDevaition();
		osw.write("\r\n------------------------Average-----------------------------\r\n");
		for (int i = 0; i < methodList.size(); i++)
			methodList.get(i).writeMean(osw);
		osw.write("\r\n------------------------Std Deviation-----------------------\r\n");
		for (int i = 0; i < methodList.size(); i++)
			methodList.get(i).writeSd(osw);
		osw.close();
	}
	
	public void addStudentTTest(String method1, String method2, double pValue) {
		String id = new String(method1+method2);
		StudentTTest sttest = Misc.getElementById(studentTTestList, id);
		
		if (sttest == null) {
			sttest = new StudentTTest(method1, method2, pValue);
			studentTTestList.add(sttest);
		}
		MethodMetrics m1 = Misc.getElementById(methodList, method1);
		MethodMetrics m2 = Misc.getElementById(methodList, method2);
		sttest.doTest(m1.metricsTable, m2.metricsTable, StudentTTest.METRIC_ACC);
		sttest.doTest(m1.metricsTable, m2.metricsTable, StudentTTest.METRIC_AUC);
	}
	
	public void printStudentTTestResults(String outPath) {
		StringBuffer outPathBuffer = new StringBuffer();
		if (outPath == null) {
			outPathBuffer.append(outDir+name+useParameterString+"-");
			Date date = new Date();
			SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd'-'HH'h'mm'm'ss's'");
			String dateStr = sdf.format(date);
			outPathBuffer.append(dateStr + ".StudentTTest.txt");
			
		} else {
			outPathBuffer.append(outPath);
		}
		
		try {
			FileOutputStream fos = new FileOutputStream(new File(outPathBuffer.toString()));
			streamOutStudentTTest(fos);
			fos.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void streamOutStudentTTest(OutputStream outStream) throws IOException {
		OutputStreamWriter osw = new OutputStreamWriter(outStream);
		osw.write("------------------------------------------------------------\r\n");
		Date date = new Date();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd't'HH'h'mm'm'ss's'");
		String dataStr = sdf.format(date);
		osw.write(name + "  " + dataStr +"\r\n");
		osw.write("Parameters: " + useParameterString + "\r\n");
		osw.write("Total Round Number: " + maxRound + "\r\n");
		osw.write("------------------------------------------------------------\r\n");
		for (int i  = 0; i < studentTTestList.size(); i++) {
			StudentTTest ttest = studentTTestList.get(i);
			osw.write("------------Method:" + ttest.getMethodA() + " V.S. Method:" + ttest.getMethodB() +"--------\r\n");
			ttest.write(osw, StudentTTest.METRIC_ACC);
			ttest.write(osw, StudentTTest.METRIC_AUC);
		}
		osw.close();
	}
	
	private void writeInstantFile(String methodName, int round, Metrics metrics) {
		if (instantWriter != null) {
			try {
				String accstr = String.format("%.4f", metrics.accuracy);
				String aucstr = String.format("%.4f", metrics.auc);
				instantWriter.write(methodName + ", " + round + ", " + accstr + ", " + aucstr+ "\r\n");
				instantWriter.flush();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	private String outDir = null;
	private String name = null;
	private String useParameterString = null;
	private int    maxRound = 0; // max round of experiments
	private ArrayList<MethodMetrics> methodList = new ArrayList<MethodMetrics>();
	
	private OutputStreamWriter instantWriter = null;
	private StringBuffer instantFilePathBuffer = null;
	
	private ArrayList<StudentTTest> studentTTestList = new ArrayList<StudentTTest>();
}
