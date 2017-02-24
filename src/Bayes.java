//package prediction;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.concurrent.ThreadLocalRandom;

public class Bayes{
	
	private static void predictClassBayes(boolean partitionTrainSet, String trainSetPath, String validationSetPath, boolean headings, int [] columnsToUse, int predictionColumn, int [] splitPoints) {
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
		
		//convert trimmed columns back to row structure
		trainSet = CommonMethods.createRowStructure(trainColumns);
		validationSet = CommonMethods.createRowStructure(validationColumns);
		
		//split classification column into sets if needed, for continuous variables
		trainSet = CommonMethods.classifyDataset(trainSet, splitPoints);
		validationSet = CommonMethods.classifyDataset(validationSet, splitPoints);
		
		trainColumns = CommonMethods.createColumnStructure(trainSet);
		validationColumns = CommonMethods.createColumnStructure(validationSet);
		
		HashMap<Double,ArrayList<Integer>> classIndexMap = getClassLocationMap(trainColumns);
		
		HashMap<Double,ArrayList<ArrayList<Double>>> classColumnStructuresMap = getClassColumnStructuresMap(trainSet, classIndexMap);
		
		double totalSize = (double)trainSet.size();
		
		double rightCount = 0;
		int g = 0;
		Iterator<ArrayList<Double>> iter6 = validationSet.iterator();
		
		while (iter6.hasNext()) { // classify validation set one by one
		
			ArrayList<Double> currentRow = iter6.next(); // current validation row
			
			HashMap<Double,Double> classProbs = new HashMap<Double,Double>();
			
			for(Double currentClass: classColumnStructuresMap.keySet()){ //for each class
				ArrayList<ArrayList<Double>> classColumnSet = classColumnStructuresMap.get(currentClass);
				double classSize = (double)classColumnSet.get(0).size();
				
				double prob = (classSize/totalSize); //prior prob
				
				for(int i = 0; i < currentRow.size()-1; i++){ //each conditional prob
					Double compValue = currentRow.get(i);
					ArrayList<Double> classDimensionColumn = classColumnSet.get(i);
					double count = (double)getColumnCount(classDimensionColumn, compValue);
					double conditionalProb = (count/classSize);
					prob *= conditionalProb;
				}

				classProbs.put(currentClass, prob);
			}
			
			Double classValue = getMaxProb(classProbs);
			if(CommonMethods.equalsDouble(currentRow.get(currentRow.size()-1),classValue))
				rightCount++;

			g++;
		}
		
		double success = Double.parseDouble(df.format((rightCount/g)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of " + g + " correct");
		System.out.println("Bayes classifier built with " + success + "% accuracy");
	
	}
	
	static HashMap<Double,ArrayList<ArrayList<Double>>> getClassColumnStructuresMap(ArrayList<ArrayList<Double>> trainSet, HashMap<Double,ArrayList<Integer>> classLocationMap){
		HashMap<Double,ArrayList<ArrayList<Double>>> classColumnStructures = new HashMap<Double,ArrayList<ArrayList<Double>>>();
		for(Double x: classLocationMap.keySet()){ //for each class
			ArrayList<ArrayList<Double>> currentClassRowStructure = new ArrayList<ArrayList<Double>>();
			for(Integer a: classLocationMap.get(x)){ //for each row number associated with this class
				currentClassRowStructure.add(trainSet.get(a));
			}
			ArrayList<ArrayList<Double>> currentClassColumnStructure = CommonMethods.createColumnStructure(currentClassRowStructure); //convert to column format
			classColumnStructures.put(x,currentClassColumnStructure); //put class, column struct in result
		}
		return classColumnStructures;
	}
	
	static HashMap<Double,ArrayList<Integer>> getClassLocationMap(ArrayList<ArrayList<Double>> trainColumns){
		ArrayList<Double> classColumn = trainColumns.get(trainColumns.size()-1); //last column
		
		HashMap<Double,ArrayList<Integer>> indexMap = new HashMap<Double,ArrayList<Integer>>();
		//ArrayList<Double> result = new ArrayList<Double>();
		
		for(int i=0; i < classColumn.size(); i++){
			Double value = classColumn.get(i);
			if(indexMap.containsKey(value)){
					ArrayList<Integer> current = indexMap.get(value);
					current.add(i); //add this index to the list of indexes corresponding to this class value
					indexMap.put(value, current);
			}
			else{
				ArrayList<Integer> current = new ArrayList<Integer>();
				current.add(i);
				indexMap.put(value, current);
			}
		}
		
		return indexMap;
	}
	
	static int getColumnCount(ArrayList<Double> list, Double value){
		int count = 0;
		for(Double d: list){
				if(CommonMethods.equalsDouble(d,value)){
					count++;
				}
		}
		return count;
	}
	
	
	private static Double getMaxProb(HashMap<Double,Double> probCounts){
		//create tree map with comparitor which sorts by value instead of key
		TreeSet<Map.Entry<Double, Double>> entriesSet = new TreeSet<>(new Comparator<Map.Entry<Double, Double>>(){
           @Override 
			public int compare(Map.Entry<Double, Double> x, Map.Entry<Double, Double> y) {
				return x.getValue().compareTo(y.getValue());
			}
        });
        entriesSet.addAll(probCounts.entrySet());
		return entriesSet.last().getKey();
	}
	
	
	public static void main(String [] args) {
	
		predictClassBayes(true, "student-mat-normalised.csv", null, true, new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18},32, new int[]{7,14});
		
		
	}
	
}


























