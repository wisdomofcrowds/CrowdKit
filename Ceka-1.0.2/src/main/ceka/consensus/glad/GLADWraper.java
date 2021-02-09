/**
 * 
 */
package ceka.consensus.glad;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import ceka.converters.GLADConverter;
import ceka.core.Dataset;
import ceka.utils.Misc;

/**
 * This class wrap the GLAD algorithm. GLAD is an exe file. <br>
 * <br> Article:<br>
 * Jacob Whitehill, Ting-fan Wu, Jacob Bergsma, Javier R Movellan, and Paul L Ruvolo.
Whose vote should count more: Optimal integration of labels from labelers of unknown
expertise. In Advances in neural information processing systems, pages 2035-2043, 2009.
 * @author Zhang
 *
 */
public class GLADWraper {
	
	public static String NAME = "GLAD";
	
	public GLADWraper (String outputDir, String gladPath) {
		this.saveDir = new String(outputDir);
		Misc.createDirectory(this.saveDir);
		if (gladPath != null) {
			gladExe.append(gladPath);
		} else {
			String currDir = System.getProperty("user.dir");
			gladExe.append(currDir + "\\tools\\GLAD.exe");
		}
		analyzer = new GLADResultsAnalyzer(NAME, threshold);
	}

	public void doInference(Dataset dataset) {
		
		String relationName = dataset.relationName();
		String gladInputPath = saveDir + relationName + ".glad.in.txt";
		String galdOutputPath = saveDir + relationName + ".glad.out.txt";
		
		Process process = null;
		try {
			GLADConverter converter = new GLADConverter();
			converter.saveDataset(dataset, gladInputPath);
			gladExe.append(" " + gladInputPath + " " + galdOutputPath);
			process = Runtime.getRuntime().exec(gladExe.toString());
			WatchThread wt = new WatchThread(process);
			wt.start();
			process.waitFor();
			wt.setOver(true);
			analyzer.analyze(galdOutputPath, converter);
			analyzer.assignIntegratedLabel(dataset);
		} catch (Exception e) {
			try {
				process.getOutputStream().close();
				process.getInputStream().close();
				process.getErrorStream().close();
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			e.printStackTrace();
		}
		// this is important
		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	class WatchThread extends Thread {
	    Process p;
	    boolean over;
	    public WatchThread(Process p) {
	        this.p = p;
	        over = false;
	    }

	    public void run() {
	        try {
	            if (p == null) return;
	            BufferedReader br = new BufferedReader(new  InputStreamReader(p.getInputStream()));
	            while (true) {
	                if (p==null || over) {
	                    break;
	                }
	                while(br.readLine()!=null);
	            }
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	    }
	    public void setOver(boolean over) {
	        this.over = over;
	    }
	}
	
	private double threshold = 0.5;
	private GLADResultsAnalyzer analyzer = null;
	private String saveDir = null;
	private StringBuffer gladExe = new StringBuffer();
}
