//package prediction;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.concurrent.ThreadLocalRandom;


//Rocchio classifier
public class NearestCentroid{
	
	private static void predictClassNearestCentroid(boolean partitionTrainSet, String trainSetPath, String validationSetPath, boolean headings, int [] columnsToUse, int predictionColumn, int [] splitPoints, String formula ) {
		
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
		//System.out.println("Training size: " + trainSet.size());
		//System.out.println("Validation size: " + validationSet.size());
		
		//create column structure
		ArrayList<ArrayList<Double>> trainColumns = CommonMethods.createColumnStructure(trainSet);
		ArrayList<ArrayList<Double>> validationColumns = CommonMethods.createColumnStructure(validationSet);
		
		//trim columns which wont be used
		trainColumns = CommonMethods.trimColumns(trainColumns, columnsToUse, predictionColumn);
		validationColumns = CommonMethods.trimColumns(validationColumns, columnsToUse, predictionColumn);
		
		//convert trimmed columns back to row structure
		trainSet = CommonMethods.createRowStructure(trainColumns);
		validationSet = CommonMethods.createRowStructure(validationColumns);
		
		//normalise set
		trainSet = CommonMethods.normaliseDataset(trainSet, trainColumns);
		validationSet = CommonMethods.normaliseDataset(validationSet, validationColumns);
		
		//split classification column into sets if needed, for continuous variables
		trainSet = CommonMethods.classifyDataset(trainSet, splitPoints);
		validationSet = CommonMethods.classifyDataset(validationSet, splitPoints);
		
		trainColumns = CommonMethods.createColumnStructure(trainSet);
		validationColumns = CommonMethods.createColumnStructure(validationSet);
		
		HashMap<Double, ArrayList<Double>> classCentroids = getClassCentroids(trainColumns);
		
		/*
		for(Double a: classCentroids.keySet()){
			System.out.println(a + ": " + classCentroids.get(a));
		}
		*/
		
		Iterator<ArrayList<Double>> iter6 = validationSet.iterator();
		
		double rightCount = 0;
		int g = 0;
		while (iter6.hasNext()) { // classify validation set one by one using knn and add to resultsMap
		
			ArrayList<Double> current = iter6.next(); // current validation row
			
			HashMap<Double,Double> centroidDistances = new HashMap<Double,Double>();
			
			for(Double currentClass: classCentroids.keySet()){
				ArrayList<Double> currentCentroid = classCentroids.get(currentClass);
				double dist = CommonMethods.calcDistance(current,currentCentroid,formula);
				centroidDistances.put(currentClass,dist);
			}
			
			Double classValue = CommonMethods.getMinCount(centroidDistances);
			
			if(CommonMethods.equalsDouble(current.get(current.size()-1),classValue))
				rightCount++;

			g++;
		}
		
		DecimalFormat df = new DecimalFormat("#.####");
		double success = Double.parseDouble(df.format((rightCount/g)*100)); // percentage success rounded to 4 places
		System.out.println(rightCount + " predictions out of " + g + " correct");
		System.out.println("Nearest centroid classifier built with " + success + "% accuracy");
		// System.out.println("total " + g);
	}
	
	private static HashMap<Double, ArrayList<Double>> getClassCentroids(ArrayList<ArrayList<Double>> trainColumns){
		HashMap<Double,ArrayList<Integer>> classLocations = CommonMethods.getColumnValueLocationMap(trainColumns, trainColumns.size()-1);
		
		HashMap<Double,ArrayList<ArrayList<Double>>> classColumnStructures = CommonMethods.getIndexColumnStructuresMap(CommonMethods.createRowStructure(trainColumns), classLocations);
		
		HashMap<Double, ArrayList<Double>> classCentroids = new HashMap<Double, ArrayList<Double>>();
		
		for(Double current: classColumnStructures.keySet()){ //for each class
			ArrayList<ArrayList<Double>> currentClassColumns = classColumnStructures.get(current);
			ArrayList<Double> currentCentroid = new ArrayList<Double>();
			for(int i =0; i < currentClassColumns.size()-1; i++){ //for each row of this class except pred column
				ArrayList<Double> currentAttribute = currentClassColumns.get(i);
				Double avg = getAvg(currentAttribute); //do this method 
				currentCentroid.add(avg);
			}
			classCentroids.put(current,currentCentroid);
		}
		return classCentroids;
	}
	
	private static double getAvg(ArrayList<Double> currentAttribute){
		double size = (double) currentAttribute.size();
		double total = 0.0;
		for(Double x: currentAttribute){
			total += x;
		}
		return (total/size);
	}
	
	public static void main(String [] args) {
	
		System.out.println("///////////// Portugese student results /////////////");
		predictClassNearestCentroid(true, "student-mat-normalised.csv", null, true, new int[]{5,7,8,9,10,11,12,13,14,15,16,28},32, new int[]{7,14}, "euclidean");
		System.out.println();
		System.out.println("///////////// Iris flower classification /////////////");
		predictClassNearestCentroid(true, "IrisNumerical.csv", null, true, new int[]{1,2,3,4},5, null, "euclidean");
		System.out.println();
		System.out.println("///////////// Car approval level /////////////");
		predictClassNearestCentroid(true, "CarDataNumerical.csv", null, true, new int[]{0,1,2,3,4,5},6, null,"euclidean");
		//{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
		
	}
}