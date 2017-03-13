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
		
		ArrayList<Integer> usedAtts = new ArrayList<Integer>();
		
		Node root = new Node(trainColumns, usedAtts);
		
		buildTree(root, formula);

		//preOrderPrint(root,0);
		
		Iterator<ArrayList<Double>> iter6 = validationSet.iterator();
		double rightCount = 0;
		int g = 0;
		
		while (iter6.hasNext()) { // classify validation set one by one using tree and add to resultsMap
			ArrayList<Double> current = iter6.next();
			double predictedClass = traverseTree(current, root);
			if(CommonMethods.equalsDouble(current.get(current.size()-1),predictedClass))
				rightCount++;
			g++;
		}
		
		double success = Double.parseDouble(df.format((rightCount/g)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of " + g + " correct");
		System.out.println("Decision tree classifier built with " + success + "% accuracy");
	}
	
	public static void preOrderPrint(Node root, int depth) {  
			if(root == null)
				return;
			
			for(int i = 0; i < depth; i++){
				System.out.print("  ");
			}
			System.out.print(root.getSplitIndex());
			
			if(root.isLeaf()){
				System.out.println("{}");
				return;
			}
			int newDepth = depth + 1;
			System.out.println("{");
			for(Double x: root.children.keySet()){
				preOrderPrint(root.children.get(x),newDepth);
			}
			for(int i = 0; i < depth; i++){
				System.out.print("  ");
			}
			System.out.println("}");
	}
		
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
			}
			columnEntropy += (valueRatio * accum);
		}
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
			double calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
			splitInfo += calc;
		}
		double gainRatio;
		if(infoGain != 0.0)
			gainRatio = (infoGain/splitInfo);
		else
			gainRatio = 0.0;
		return gainRatio;
	}
	
	static double getEntropy(ArrayList<ArrayList<Double>> trainColumns, int columnIndex){
		HashMap<Double,ArrayList<Integer>> classIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, columnIndex);  //hash map: key is class, value is arraylist of indexes
		double setEntropy = 0.0;
		double setSize = (double)trainColumns.get(columnIndex).size();
		for(Double currentClass: classIndexMap.keySet()){
			double size = (double)classIndexMap.get(currentClass).size();
			double ratio = (size/setSize);
			double calc = (-1 * ratio * (Math.log(ratio)/Math.log(2)));
			setEntropy += calc;
		}
		return setEntropy;
	}
	
	static double gainFormula(ArrayList<ArrayList<Double>> trainColumns, int targetColumnIndex, String formula){
		if(formula.equals("GainRatio")){
			return gainRatio(trainColumns,targetColumnIndex);
		}
		else{ //information gain
			return infoGain(trainColumns,targetColumnIndex);
		}
	}
	
	static int selectAttribute(ArrayList<ArrayList<Double>> trainColumns, String formula, ArrayList<Integer> usedAttributes){ //returns index of attribute to be used for next split
		int bestIndex = -1;
		double bestGain = 0.0;
		for(int i = 0; i < trainColumns.size()-1; i++){ //last element is pred column
			if(!usedAttributes.contains(i)){
				double currentGain = gainFormula(trainColumns,i,formula);
				if(currentGain >= bestGain){
					bestGain = currentGain;
					bestIndex = i;
				}
			}		
		}
		return bestIndex;
	}
	
	static Node buildTree(Node root, String formula) {
		//check if leaf
			//1 class left - entropy will be 0
			ArrayList<ArrayList<Double>> currentColumns = root.getData();
			Double entropy = getEntropy(currentColumns, (currentColumns.size()-1));
			if(CommonMethods.equalsDouble(entropy,0.0)){
				root.setLeaf(true);
				return root;
			}
		
			//fully split - usedAttributes will be = to columns.size -1
			
			if(root.getUsedAttributes().size() == (currentColumns.size()-1)){
				root.setLeaf(true);
				return root;
			}
			
		//if not - split & get children and call on them
		
		int bestAttribute = selectAttribute(currentColumns,formula, root.getUsedAttributes());
		
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(currentColumns, bestAttribute);
		
		HashMap<Double,ArrayList<ArrayList<Double>>> indexColumnStructuresMap = CommonMethods.getIndexColumnStructuresMap(CommonMethods.createRowStructure(currentColumns), columnValueIndexMap);
		
		root.setSplitIndex(bestAttribute);
		
		ArrayList<Integer> atts = root.getUsedAttributes();
		atts.add(bestAttribute);
		
		for (Double a: indexColumnStructuresMap.keySet()) {
			Node currentChild = new Node(indexColumnStructuresMap.get(a),atts);
			root.children.put(a,currentChild);
		}
		
		for(Double x: root.children.keySet()){
			buildTree(root.children.get(x),formula);
		}
		
		return root;	
	}
	
	static Double traverseTree(ArrayList<Double> r, Node root) {
		
		if(!root.isLeaf()){ //stop if current node is leaf
			double nodeValue = r.get(root.getSplitIndex()); //the value of the index this node splits on
			
			if(root.children.containsKey(nodeValue)){
				return traverseTree(r,root.children.get(nodeValue));
			}
			 //if false -this means this value is not in children - so stop at current node
		}
	
		//tree has been traversed, get majority count of root data
		ArrayList<ArrayList<Double>> resultTrainColumns = root.getData();
		HashMap<Double,ArrayList<Integer>> columnValueIndexMap = CommonMethods.getColumnValueLocationMap(resultTrainColumns, (resultTrainColumns.size()-1));
		HashMap<Double,Double> classCounts = new HashMap<Double,Double>();
		for(Double x: columnValueIndexMap.keySet()){
			double size = (double)columnValueIndexMap.get(x).size();
			classCounts.put(x,size);
		}
		double maxClass = CommonMethods.getMaxCount(classCounts);
		return maxClass;
	}
	
	public static void main(String [] args) {
	
		predictClassDecisionTree(true, "student-mat-normalised.csv", null, true, new int[]{0,1,15,17,18,26},32, new int[]{7,14},"GainRatio");
		predictClassDecisionTree(true, "IrisNumerical.csv", null, true, new int[]{1,2,3,4},5, null, "Gain Ratio");
		predictClassDecisionTree(true, "CarDataNumerical.csv", null, true, new int[]{0,1,2,3,4,5},6, null, "GainRatio");
		//predictClassDecisionTree(true, "CarDataNumerical.csv", null, true, new int[]{0,1,2,3,4,5},6, null, "Information Gain");
		
	}	
}

class Node {
	public HashMap<Double, Node> children;
	private ArrayList<ArrayList<Double>> trainColumns;
	private ArrayList<Integer> usedAttributes;
	private int splitIndex;
	boolean isLeaf;
	
	public Node(ArrayList<ArrayList<Double>> trainColumns, ArrayList<Integer> usedAttributes) {
		ArrayList<Integer> copy = new ArrayList<Integer>(usedAttributes); //to stop pass by reference
		children = new HashMap<Double, Node>();
		this.trainColumns = trainColumns;
		this.usedAttributes = copy;
		setSplitIndex(-1);
		setLeaf(false);
	}
	
	public ArrayList<Integer> getUsedAttributes(){
		return this.usedAttributes;
	}

	public ArrayList<ArrayList<Double>> getData() {
		return trainColumns;
	}

	public void setLeaf(boolean isLeaf) {
		this.isLeaf = isLeaf;
	}

	public boolean isLeaf() {
		return isLeaf;
	}
	
	public void setSplitIndex(int splitIndex) {
		this.splitIndex = splitIndex;
	}

	public int getSplitIndex() {
		return splitIndex;
	}
}

















