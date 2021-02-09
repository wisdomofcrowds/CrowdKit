/**
 * 
 */
package com.ipeirotis.gal.engine.rpt;

import java.io.File;
import java.io.FileWriter;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import com.ipeirotis.gal.algorithms.DawidSkene;
import com.ipeirotis.gal.core.Category;
import com.ipeirotis.gal.core.Datum;

/**
 * @author Zhang
 *
 */
public class MyReport {
	
	public static void report(DawidSkene dsAlgorithm, String resultFile) {
		try {
			FileWriter resultWriter = new FileWriter(new File(resultFile));
			boolean firstLine = true;
			
			Map<String, Datum> objects = dsAlgorithm.getObjects();
			Iterator<Entry<String, Datum>> objsIter = objects.entrySet().iterator();
			while (objsIter.hasNext()) {
				Entry<String, Datum> entryObj = objsIter.next();
				String objId = entryObj.getKey();
				Datum obj = entryObj.getValue();
				String label = obj.getSingleClassClassification(Datum.ClassificationMethod.DS_MaxLikelihood);
				if (firstLine) {
					resultWriter.write(objId + "\t" + label);
					firstLine = false;
				}
				else {
					resultWriter.write("\n" + objId + "\t" + label);
				}
			}
			resultWriter.close();
			
			Map<String, Category> categories = dsAlgorithm.getCategories();
			for (String c : categories.keySet()) {
				Category category = categories.get(c);
				Double prior = category.getPrior();
				System.out.println("Category: " + c + ":  " + prior);
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
}
