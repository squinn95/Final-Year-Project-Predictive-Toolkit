//package prediction;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.concurrent.ThreadLocalRandom;

public class DecisionTree{
	
	private static void predictClassDecisionTree(boolean partitionTrainSet, String trainSetPath, String validationSetPath, boolean headings, int [] columnsToUse, int predictionColumn, int [] splitPoints, String formula) {
		DecimalFormat df = new DecimalFormat("#.####");
		// read train 
		
		ArrayList<ArrayList<Double>> trainSet = CommonMethods.readDatasetFile(trainSetPath, headings);
		ArrayList<ArrayList<Double>> validationSet;
		//split train set into two if needed
		if(partitionTrainSet){
			Pair<ArrayList<ArrayList<Double>>,ArrayList<ArrayList<Double>>> splits = CommonMethods.randomlyHalfSet(trainSet);
			trainSet = splits.getFirst();
			validationSet = splits.getSecond();
		}
		else{
			validationSet = CommonMethods.readDatasetFile(validationSetPath, headings);
		}
		
		//create column structure
		ArrayList<ArrayList<Double>> trainColumns = CommonMethods.createColumnStructure(trainSet);
		ArrayList<ArrayList<Double>> validationColumns = CommonMethods.createColumnStructure(validationSet);
		
		//trim columns which wont be used
		trainColumns = CommonMethods.trimColumns(trainColumns, columnsToUse, predictionColumn);
		validationColumns = CommonMethods.trimColumns(validationColumns, columnsToUse, predictionColumn);
		
		//update row structure
		trainSet = CommonMethods.createRowStructure(trainColumns);
		validationSet = CommonMethods.createRowStructure(validationColumns);
		
		//split classification column into sets if needed, for continuous variables
		trainSet = CommonMethods.classifyDataset(trainSet, splitPoints);
		validationSet = CommonMethods.classifyDataset(validationSet, splitPoints);
		
		//update column structure
		trainColumns = CommonMethods.createColumnStructure(trainSet);
		validationColumns = CommonMethods.createColumnStructure(validationSet);
		
		
		for(int i =0; i < trainColumns.size() -1; i++){
			//System.out.println(i + ": " + infoGain(trainColumns,i));
			gainRatio(trainColumns,i);
		}
		
		
		ArrayList<Integer> usedAttributes = new ArrayList<Integer>();
		Node root = new Node();
		root.setData(trainColumns);
		root.setUsedAttributes(new ArrayList<Integer>());
		
		buildTree(root, formula);
		
		//traverseTree(test,root);
		
		//preOrder(root);
		
		
		
		
		
		
		ArrayList<Double> test = new ArrayList<Double>();
		test.add(new Double(1.0));
		test.add(new Double(1.0));
		test.add(new Double(2.0));
		test.add(new Double(2.0));
		test.add(new Double(2.0));
		test.add(new Double(2.0));
		test.add(new Double(1.0));
		
		traverseTree(test, root);
		
		//System.out.println(classResult);
		
		
		
		
		
		/*
		double rightCount = 0;
		int g = 0;
		Iterator<ArrayList<Double>> iter6 = validationSet.iterator();
		
		while (iter6.hasNext()) { // classify validation set one by one
		
			ArrayList<Double> currentRow = iter6.next(); // current validation row
			
			//classify and increment right count

			g++;
		}
		
		double success = Double.parseDouble(df.format((rightCount/g)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of " + g + " correct");
		System.out.println("Bayes classifier built with " + success + "% accuracy");
		*/
	}
	
	//Methods
	
	//gain ratio: args column, pred column
	
	/*
	preorder(node)
  if (node = null)
    return
  visit(node)
  preorder(node.left)
  preorder(node.right)
	*/
	
	
	
	static double infoGain(ArrayList<ArrayList<Double>> trainColumns, int targetColumnIndex){ //pred column last index
	
		HashMap<Double,ArrayList<Integer>> classIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, trainColumns.size()-1);  //hash map: key is class, value is arraylist of indexes
		
		double setEntropy = getEntropy(trainColumns, trainColumns.size()-1);
		double setSize = (double)trainColumns.get(0).size();
		
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, targetColumnIndex);
		
		double columnEntropy = 0.0;
		for(Double currentColumnValue: columnValueIndexMap.keySet()){
			double valueCount = (double)columnValueIndexMap.get(currentColumnValue).size();
			double valueRatio = (valueCount/setSize);
			
			double accum = 0.0;
			for(Double currentClass: classIndexMap.keySet()){
				double commonCount = 0.0;
				for(Integer index: classIndexMap.get(currentClass)){
					if(columnValueIndexMap.get(currentColumnValue).contains(index)){
						commonCount++;
					}
				}
			
				double ratio = (commonCount/valueCount);
				double calc;
				if (ratio > 0.0)
					calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
				else
					calc = 0.0;
				
				accum += calc;
				//System.out.println("columnValue: " + currentColumnValue + ", currentColumnValue size: " + valueCount + ", class: " + currentClass + ", count: " + commonCount + ", calc: " + calc);
			}
			columnEntropy += (valueRatio * accum);
			//System.out.println();
		}
		
		//System.out.println("Set entropy: " +setEntropy + ", Column entropy: " + columnEntropy);
		
		double gain = (setEntropy - columnEntropy);
		
		return gain;
	}
	
	static double gainRatio(ArrayList<ArrayList<Double>> trainColumns, int targetColumnIndex){
		
		double infoGain = infoGain(trainColumns, targetColumnIndex);
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, targetColumnIndex);
		double setSize = (double)trainColumns.get(0).size();
		
		double splitInfo = 0.0;
		for(Double currentClass: columnValueIndexMap.keySet()){
			double size = (double)columnValueIndexMap.get(currentClass).size();
			double ratio = (size/setSize);
			double calc;
				if (ratio > 0.0)
					calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
				else
					calc = 0.0;
			splitInfo += calc;
		}
		
		double gainRatio = (infoGain/splitInfo);
		//System.out.println("info gain: " + infoGain + ", gainRatio: " + gainRatio);
		return gainRatio;
		
	}
	
	static double getEntropy(ArrayList<ArrayList<Double>> trainColumns, int columnIndex){
		HashMap<Double,ArrayList<Integer>> classIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, columnIndex);  //hash map: key is class, value is arraylist of indexes
		
		double setEntropy = 0.0;
		double setSize = (double)trainColumns.get(columnIndex).size();
		for(Double currentClass: classIndexMap.keySet()){
			double size = (double)classIndexMap.get(currentClass).size();
			double ratio = (size/setSize);
			double calc;
				if (ratio > 0.0){
					calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
					if(calc == 0.0){
						System.out.println("proper 0 entropy");
					}
				}
				else
					calc = 0.0;
			setEntropy += calc;
			//System.out.println("class: " + currentClass + ", size: " + size + "/" + setSize + ", ratio: " + ratio + ", Entropy: " + calc);
		}
		return setEntropy;
	}
	
	
	
	static int selectAttribute(ArrayList<ArrayList<Double>> trainColumns, String formula, ArrayList<Integer> usedAttributes){ //returns index of attribute to be used for next split
		
		if(trainColumns.size() > 1){
			if(formula.equals("Gain Ratio")){
				int bestIndex = -1;
				double bestGR = 0.0;
				for(int i = 0; i < trainColumns.size()-1; i++){ //last element is pred column
					if(!usedAttributes.contains(i)){
						double currentGR = gainRatio(trainColumns,i);
						//System.out.print("row: " + i + ", GR: " + currentGR + " /// ");
						if(currentGR > bestGR){
							bestGR = currentGR;
							bestIndex = i;
						}
					}
					
				}
				//System.out.println();
				return bestIndex;
			}
			else{ //information gain
				int bestIndex = -1;
				double bestIG = 0.0;
				for(int i = 0; i < trainColumns.size()-1; i++){ //last element is pred column
					if(!usedAttributes.contains(i)){
						double currentIG = infoGain(trainColumns,i);
						if(currentIG > bestIG){
							bestIG = currentIG;
							bestIndex = i;
						}
					}
	
				}
				return bestIndex;
			}
		}
		else{
			
			return -1;
			
		}
	}
	
	public static void preOrder(Node root) {  
			if(root !=  null) {  
			   //Visit the node-Printing the node data    
				  System.out.print(root.getUsedAttributes().size() + "{");  
				  //preorder(root.left);  
				  //preorder(root.right); 
				if((root.getChildren() != null)){
					for(Node child: root.getChildren()){
						int count = 0;
						Node copy = root;
						while((copy = copy.getParent()) != null)
							count++;
						System.out.println();
						for(int i =-1;i < count;i++)
							System.out.print('\t');
						preOrder(child);
						System.out.println();
						for(int i =-0;i < count;i++)
							System.out.print('\t');
					}
					System.out.print("}");
				}
				else{
					System.out.print("}");
				} 
			}  
	}
	
	static Node buildTree(Node root, String formula) { //formula = "Information Gain"

		//root.setEntropy(getEntropy(root.getData(),(root.getData.size() -1))); //root contains trainColumns, last index is pred column
	
		double classEntropy = getEntropy(root.getData(),(root.getData().size()-1));
		//System.out.println("classEntropy: " + classEntropy);
		if(CommonMethods.equalsDouble(classEntropy, new Double(0.0))){
			//System.out.println("classEntropy: " + classEntropy + " - returned root");
			return root;
		}
	
		
		int bestAttribute = selectAttribute(root.getData(),formula, root.getUsedAttributes()); //this works
		//System.out.println("classEntropy: " + classEntropy + ", Best attribute index: " + bestAttribute);
		if(bestAttribute != -1) {
			
			HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(root.getData(), bestAttribute);
			
			int setSize = columnValueIndexMap.keySet().size(); //how many children has this attribute

			root.children = new Node[setSize]; //node array of size setSize
			
			HashMap<Double,ArrayList<ArrayList<Double>>> classColumnStructuresMap = CommonMethods.getClassColumnStructuresMap(CommonMethods.createRowStructure(root.getData()), columnValueIndexMap);
			
			ArrayList<Integer> atts = root.getUsedAttributes();
			atts.add(bestAttribute);
			
			int j = 0;
			for (Double a: classColumnStructuresMap.keySet()) {
				root.children[j] = new Node();
				root.children[j].setParent(root);
				root.children[j].setData(classColumnStructuresMap.get(a));
				root.children[j].setUsedAttributes(atts);
				root.children[j].setParentSplitIndex(bestAttribute);
				root.children[j].setParentSplitValue(a); //value of split from parent
				//root.children[j].name = root.name;
				j++;
			}

			System.out.println(root.getUsedAttributes());
			
			for (int k = 0; k < setSize; k++) {
				//ArrayList<Integer> usedAttributesCopy = usedAttributes; //so that one branch of tree dosnt stop other branches using a particular attribute
				//buildTree(root.children[k], formula, usedAttributesCopy);
				buildTree(root.children[k], formula);
			}

			//root.setData(null); //not sure why you would do this
		}
		
		/*
		if(bestAttribute != -1) {
			int setSize = Hw1.setSize(Hw1.attrMap.get(bestAttribute));
			root.setTestAttribute(new DiscreteAttribute(Hw1.attrMap.get(bestAttribute), 0));
			root.children = new Node[setSize];
			root.setUsed(true);
			Hw1.usedAttributes.add(bestAttribute);
			
			for (int j = 0; j< setSize; j++) {
				root.children[j] = new Node();
				root.children[j].setParent(root);
				root.children[j].setData(subset(root, bestAttribute, j));
				root.children[j].getTestAttribute().setName(Hw1.getLeafNames(bestAttribute, j));
				root.children[j].getTestAttribute().setValue(j);
			}

			for (int j = 0; j < setSize; j++) {
				buildTree(root.children[j].getData(), root.children[j], learningSet);
			}

			root.setData(null);
		}
		
		*/
		else {
			return root;
		}
		
		return root;
	}
	
	
	static void traverseTree(ArrayList<Double> r, Node root) {
		while(root.children != null) {
			double nodeValue = r.get(root.getSplitIndex()); //the value of the index this node splits on
			//for(int i = 0; i < r.getAttributes().size(); i++) {
			
			for(int i = 0; i < root.getChildren().length; i++) {
				//if(nodeValue == root.children[i].getTestAttribute().getValue()) {
				if(nodeValue == root.children[i].getParentSplitValue()) {
					traverseTree(r, root.children[i]);
				}
			}
		}
		
		System.out.print("Prediction for input: ");
		 
		//tree has been traversed, get majority count of root data
		ArrayList<ArrayList<Double>> rootSet = root.getData();
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(rootSet, (rootSet.size()-1));
		HashMap<Double,Double> classCounts = new HashMap<Double,Double>();
		for(Double x: columnValueIndexMap.keySet()){
			double size = (double)columnValueIndexMap.get(x).size();
			classCounts.put(x,size);
		}
		double maxClass = getMaxCount(classCounts);
		return;
	}
	
	
	private static Double getMaxCount(HashMap<Double,Double> classCounts){
		//create tree map with comparitor which sorts by value instead of key
		TreeSet<Map.Entry<Double, Double>> entriesSet = new TreeSet<>(new Comparator<Map.Entry<Double, Double>>(){
           @Override 
			public int compare(Map.Entry<Double, Double> x, Map.Entry<Double, Double> y) {
				return x.getValue().compareTo(y.getValue());
			}
        });
        entriesSet.addAll(classCounts.entrySet());
		return entriesSet.last().getKey();
	}
	
	public static void main(String [] args) {
	
		//predictClassBayes(true, "student-mat-normalised.csv", null, true, new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18},32, new int[]{7,14});
		//predictClassBayes(true, "IrisNumerical.csv", null, true, new int[]{1,2,3,4},5, null);
		predictClassDecisionTree(true, "CarDataNumerical.csv", null, true, new int[]{0,1,2,3,4,5},6, null, "Gain Ratio");
		
		
	}
	
}

class Node {
	private Node parent;
	public Node[] children;
	public String name;
	private ArrayList<ArrayList<Double>> trainColumns;
	private ArrayList<Integer> usedAttributes;
	private boolean isUsed;
	private int parentSplitIndex;
	private double parentSplitValue;
	

	public Node() {
		trainColumns = new ArrayList<ArrayList<Double>>();
		usedAttributes = new ArrayList<Integer>();
		name = "";
		setParent(null);
		setChildren(null);
		setUsed(false);
	}

	public void setParent(Node parent) {
		this.parent = parent;
	}
	
	public void setParentSplitIndex(int attributeIndex){
		this.parentSplitIndex = attributeIndex;
	}
	
	public int getParentSplitIndex(){
		return this.parentSplitIndex;
	}
	
	public void setParentSplitValue(Double parentSplitValue){
		this.parentSplitValue = parentSplitValue;
	}
	
	public Double getParentSplitValue(){
		return this.parentSplitValue;
	}

	public Node getParent() {
		return parent;
	}
	
	public void setUsedAttributes(ArrayList<Integer> usedAttributes){
		this.usedAttributes = usedAttributes;
	}
	
	public ArrayList<Integer> getUsedAttributes(){
		return this.usedAttributes;
	}

	public void setData(ArrayList<ArrayList<Double>> trainColumns) {
		this.trainColumns = trainColumns;
	}

	public ArrayList<ArrayList<Double>> getData() {
		return trainColumns;
	}

	public void setChildren(Node[] children) {
		this.children = children;
	}

	public Node[] getChildren() {
		return children;
	}

	public void setUsed(boolean isUsed) {
		this.isUsed = isUsed;
	}

	public boolean isUsed() {
		return isUsed;
	}
}





































