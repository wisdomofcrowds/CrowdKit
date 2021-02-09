package ceka.learn.common;

import ceka.utils.IdDecorated;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class SimpleClassifier implements IdDecorated {
	
	public SimpleClassifier(String wekaClassifierName, String idStr) 
			throws ClassNotFoundException, InstantiationException, IllegalAccessException {
		id = new String(idStr);
		Class<?> m_class = Class.forName(wekaClassifierName);
		m_classifier = (Classifier) m_class.newInstance();
	}

	public void buildClassifier(Instances dataset) throws Exception {
		dataset.setClassIndex(dataset.numAttributes()-1);
		m_classifier.buildClassifier(dataset);
	}
	
	public int classifyInstance(Instance instance) throws Exception {
		return (int) m_classifier.classifyInstance(instance);
	}
	
	public Classifier getClassifier() {
		return m_classifier;
	}
	
	@Override
	public String getId() {
		// TODO Auto-generated method stub
		return id;
	}
	
	private String id = null;
	private Classifier m_classifier = null;
}
