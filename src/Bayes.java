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
		
		HashMap<Double,ArrayList<Integer>> classIndexMap = CommonMethods.getColumnValueLocationMap(trainColumns, trainColumns.size()-1);
		
		HashMap<Double,ArrayList<ArrayList<Double>>> classColumnStructuresMap = CommonMethods.getIndexColumnStructuresMap(trainSet, classIndexMap);
		
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
					double count = (double)CommonMethods.getColumnCount(classDimensionColumn, compValue);
					double conditionalProb = (count/classSize);
					prob *= conditionalProb;
				}

				classProbs.put(currentClass, prob);
			}
			
			Double classValue = CommonMethods.getMaxCount(classProbs);
			if(CommonMethods.equalsDouble(currentRow.get(currentRow.size()-1),classValue))
				rightCount++;

			g++;
		}
		
		double success = Double.parseDouble(df.format((rightCount/g)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of " + g + " correct");
		System.out.println("Bayes classifier built with " + success + "% accuracy");
	
	}	
	
	public static void main(String [] args) {
		//predictClassBayes(true, "student-mat-normalised.csv", null, true, new int[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18},32, new int[]{7,14});
		//predictClassBayes(true, "IrisNumerical.csv", null, true, new int[]{1,2,3,4},5, null);
		predictClassBayes(true, "CarDataNumerical.csv", null, true, new int[]{0,1,2,3,4,5},6, null);	
	}
}


























